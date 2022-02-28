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
    auto *inst = builder.CreateAlloca(
        moduleTranslation.convertType(op.getResult().getType())->getPointerElementType(), addrSpace, moduleTranslation.lookupValue(op.getArraySize()));
    
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
if (auto op = dyn_cast<::mlir::LLVM::AssumeOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn =
        llvm::Intrinsic::getDeclaration(module, llvm::Intrinsic::assume, {});
    builder.CreateCall(fn, {moduleTranslation.lookupValue(op.getCond())});
  
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
if (auto op = dyn_cast<::mlir::LLVM::BitReverseOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::bitreverse,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
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
if (auto op = dyn_cast<::mlir::LLVM::CopySignOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::copysign,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::CoroAlignOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::coro_align,
        { moduleTranslation.convertType(opInst.getResult(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::CoroBeginOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::coro_begin,
        { 
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::CoroEndOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::coro_end,
        { 
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::CoroFreeOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::coro_free,
        { 
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::CoroIdOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::coro_id,
        { 
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::CoroResumeOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::coro_resume,
        { 
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::CoroSaveOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::coro_save,
        { 
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::CoroSizeOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::coro_size,
        { moduleTranslation.convertType(opInst.getResult(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::CoroSuspendOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::coro_suspend,
        { 
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::CosOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::cos,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::CountLeadingZerosOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::ctlz,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::CountTrailingZerosOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::cttz,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::CtPopOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::ctpop,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::EhTypeidForOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::eh_typeid_for,
        { 
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::Exp2Op>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::exp2,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::ExpOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::exp,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
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
if (auto op = dyn_cast<::mlir::LLVM::FAbsOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::fabs,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FAddOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateFAdd(moduleTranslation.lookupValue(op.getLhs()), moduleTranslation.lookupValue(op.getRhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FCeilOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::ceil,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
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
if (auto op = dyn_cast<::mlir::LLVM::FFloorOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::floor,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FMAOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::fma,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FMulAddOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::fmuladd,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
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
    moduleTranslation.mapValue(op.getRes()) = builder.CreateGEP(
      moduleTranslation.lookupValue(op.getBase())->getType()->getPointerElementType(), moduleTranslation.lookupValue(op.getBase()), indices);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::GetActiveLaneMaskOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::get_active_lane_mask,
        { moduleTranslation.convertType(opInst.getResult(0).getType()), moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
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

    auto *inst = builder.CreateLoad(
      moduleTranslation.lookupValue(op.getAddr())->getType()->getPointerElementType(), moduleTranslation.lookupValue(op.getAddr()), op.getVolatile_());
  
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
if (auto op = dyn_cast<::mlir::LLVM::Log10Op>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::log10,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::Log2Op>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::log2,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::LogOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::log,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::MaskedLoadOp>(opInst)) {

    llvm::Type *Ty = moduleTranslation.lookupValue(op.getData())->getType()->getPointerElementType();
    moduleTranslation.mapValue(op.getRes()) = moduleTranslation.lookupValues(op.getPassThru()).empty() ? builder.CreateMaskedLoad(
      Ty, moduleTranslation.lookupValue(op.getData()), llvm::Align(op.getAlignment()), moduleTranslation.lookupValue(op.getMask())) :
      builder.CreateMaskedLoad(
        Ty, moduleTranslation.lookupValue(op.getData()), llvm::Align(op.getAlignment()), moduleTranslation.lookupValue(op.getMask()), moduleTranslation.lookupValues(op.getPassThru())[0]);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::MaskedStoreOp>(opInst)) {

    builder.CreateMaskedStore(
      moduleTranslation.lookupValue(op.getValue()), moduleTranslation.lookupValue(op.getData()), llvm::Align(op.getAlignment()), moduleTranslation.lookupValue(op.getMask()));
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::MatrixColumnMajorLoadOp>(opInst)) {

    llvm::MatrixBuilder mb(builder);
    const llvm::DataLayout &dl =
      builder.GetInsertBlock()->getModule()->getDataLayout();
    llvm::Type *ElemTy = moduleTranslation.lookupValue(op.getData())->getType()->getPointerElementType();
    llvm::Align align = dl.getABITypeAlign(ElemTy);
    moduleTranslation.mapValue(op.getRes()) = mb.CreateColumnMajorLoad(
      ElemTy, moduleTranslation.lookupValue(op.getData()), align, moduleTranslation.lookupValue(op.getStride()), op.getIsVolatile(), op.getRows(),
      op.getColumns());
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::MatrixColumnMajorStoreOp>(opInst)) {

    llvm::MatrixBuilder mb(builder);
    const llvm::DataLayout &dl =
      builder.GetInsertBlock()->getModule()->getDataLayout();
    llvm::Align align = dl.getABITypeAlign(
      moduleTranslation.lookupValue(op.getData())->getType()->getPointerElementType());
    mb.CreateColumnMajorStore(
      moduleTranslation.lookupValue(op.getMatrix()), moduleTranslation.lookupValue(op.getData()), align, moduleTranslation.lookupValue(op.getStride()), op.getIsVolatile(),
      op.getRows(), op.getColumns());
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::MatrixMultiplyOp>(opInst)) {

    llvm::MatrixBuilder mb(builder);
    moduleTranslation.mapValue(op.getRes()) = mb.CreateMatrixMultiply(
      moduleTranslation.lookupValue(op.getLhs()), moduleTranslation.lookupValue(op.getRhs()), op.getLhsRows(), op.getLhsColumns(),
      op.getRhsColumns());
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::MatrixTransposeOp>(opInst)) {

    llvm::MatrixBuilder mb(builder);
    moduleTranslation.mapValue(op.getRes()) = mb.CreateMatrixTranspose(
      moduleTranslation.lookupValue(op.getMatrix()), op.getRows(), op.getColumns());
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::MaxNumOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::maxnum,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::MaximumOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::maximum,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::MemcpyInlineOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::memcpy_inline,
        { moduleTranslation.convertType(opInst.getOperand(0).getType()), moduleTranslation.convertType(opInst.getOperand(1).getType()), moduleTranslation.convertType(opInst.getOperand(2).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::MemcpyOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::memcpy,
        { moduleTranslation.convertType(opInst.getOperand(0).getType()), moduleTranslation.convertType(opInst.getOperand(1).getType()), moduleTranslation.convertType(opInst.getOperand(2).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::MemmoveOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::memmove,
        { moduleTranslation.convertType(opInst.getOperand(0).getType()), moduleTranslation.convertType(opInst.getOperand(1).getType()), moduleTranslation.convertType(opInst.getOperand(2).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::MemsetOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::memset,
        { moduleTranslation.convertType(opInst.getOperand(0).getType()), moduleTranslation.convertType(opInst.getOperand(2).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::MinNumOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::minnum,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::MinimumOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::minimum,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
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
if (auto op = dyn_cast<::mlir::LLVM::PowIOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::powi,
        { moduleTranslation.convertType(opInst.getOperand(0).getType()), moduleTranslation.convertType(opInst.getOperand(1).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::PowOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::pow,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::Prefetch>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::prefetch,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;
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
if (auto op = dyn_cast<::mlir::LLVM::SAddWithOverflowOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::sadd_with_overflow,
        { moduleTranslation.convertType(
        opInst.getResult(0).getType().cast<LLVM::LLVMStructType>()
              .getBody()[0])
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
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
if (auto op = dyn_cast<::mlir::LLVM::SMaxOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::smax,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::SMinOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::smin,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::SMulWithOverflowOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::smul_with_overflow,
        { moduleTranslation.convertType(
        opInst.getResult(0).getType().cast<LLVM::LLVMStructType>()
              .getBody()[0])
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::SRemOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateSRem(moduleTranslation.lookupValue(op.getLhs()), moduleTranslation.lookupValue(op.getRhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::SSubWithOverflowOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::ssub_with_overflow,
        { moduleTranslation.convertType(
        opInst.getResult(0).getType().cast<LLVM::LLVMStructType>()
              .getBody()[0])
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
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
if (auto op = dyn_cast<::mlir::LLVM::SinOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::sin,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::SqrtOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::sqrt,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::StackRestoreOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::stackrestore,
        { 
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::StackSaveOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::stacksave,
        { 
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
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
if (auto op = dyn_cast<::mlir::LLVM::UAddWithOverflowOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::uadd_with_overflow,
        { moduleTranslation.convertType(
        opInst.getResult(0).getType().cast<LLVM::LLVMStructType>()
              .getBody()[0])
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
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
if (auto op = dyn_cast<::mlir::LLVM::UMaxOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::umax,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::UMinOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::umin,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::UMulWithOverflowOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::umul_with_overflow,
        { moduleTranslation.convertType(
        opInst.getResult(0).getType().cast<LLVM::LLVMStructType>()
              .getBody()[0])
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::URemOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateURem(moduleTranslation.lookupValue(op.getLhs()), moduleTranslation.lookupValue(op.getRhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::USubWithOverflowOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::usub_with_overflow,
        { moduleTranslation.convertType(
        opInst.getResult(0).getType().cast<LLVM::LLVMStructType>()
              .getBody()[0])
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
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
if (auto op = dyn_cast<::mlir::LLVM::masked_compressstore>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::masked_compressstore,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::masked_expandload>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::masked_expandload,
        { moduleTranslation.convertType(opInst.getResult(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::masked_gather>(opInst)) {

    llvm::VectorType *PtrVecTy = cast<llvm::VectorType>(moduleTranslation.lookupValue(op.getPtrs())->getType());
    llvm::Type *Ty = llvm::VectorType::get(
      PtrVecTy->getElementType()->getPointerElementType(),
      PtrVecTy->getElementCount());
    moduleTranslation.mapValue(op.getRes()) = moduleTranslation.lookupValues(op.getPassThru()).empty() ? builder.CreateMaskedGather(
      Ty, moduleTranslation.lookupValue(op.getPtrs()), llvm::Align(op.getAlignment()), moduleTranslation.lookupValue(op.getMask())) :
      builder.CreateMaskedGather(
        Ty, moduleTranslation.lookupValue(op.getPtrs()), llvm::Align(op.getAlignment()), moduleTranslation.lookupValue(op.getMask()), moduleTranslation.lookupValues(op.getPassThru())[0]);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::masked_scatter>(opInst)) {

    builder.CreateMaskedScatter(
      moduleTranslation.lookupValue(op.getValue()), moduleTranslation.lookupValue(op.getPtrs()), llvm::Align(op.getAlignment()), moduleTranslation.lookupValue(op.getMask()));
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_reduce_add>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::vector_reduce_add,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_reduce_and>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::vector_reduce_and,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_reduce_fadd>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::vector_reduce_fadd,
        { moduleTranslation.convertType(opInst.getOperand(1).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    llvm::FastMathFlags origFM = builder.getFastMathFlags();
    llvm::FastMathFlags tempFM = origFM;
    tempFM.setAllowReassoc(op.getReassoc());
    builder.setFastMathFlags(tempFM);  // set fastmath flag
    moduleTranslation.mapValue(op.getRes()) = builder.CreateCall(fn, operands);
    builder.setFastMathFlags(origFM);  // restore fastmath flag
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_reduce_fmax>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::vector_reduce_fmax,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_reduce_fmin>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::vector_reduce_fmin,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_reduce_fmul>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::vector_reduce_fmul,
        { moduleTranslation.convertType(opInst.getOperand(1).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    llvm::FastMathFlags origFM = builder.getFastMathFlags();
    llvm::FastMathFlags tempFM = origFM;
    tempFM.setAllowReassoc(op.getReassoc());
    builder.setFastMathFlags(tempFM);  // set fastmath flag
    moduleTranslation.mapValue(op.getRes()) = builder.CreateCall(fn, operands);
    builder.setFastMathFlags(origFM);  // restore fastmath flag
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_reduce_mul>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::vector_reduce_mul,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_reduce_or>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::vector_reduce_or,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_reduce_smax>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::vector_reduce_smax,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_reduce_smin>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::vector_reduce_smin,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_reduce_umax>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::vector_reduce_umax,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_reduce_umin>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::vector_reduce_umin,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_reduce_xor>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::vector_reduce_xor,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vscale>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::vscale,
        { moduleTranslation.convertType(opInst.getResult(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
