static LLVM_ATTRIBUTE_UNUSED ::llvm::InlineAsm::AsmDialect convertAsmDialectToLLVM(::mlir::LLVM::AsmDialect value) {
  switch (value) {
  case ::mlir::LLVM::AsmDialect::AD_ATT:
    return ::llvm::InlineAsm::AsmDialect::AD_ATT;
  case ::mlir::LLVM::AsmDialect::AD_Intel:
    return ::llvm::InlineAsm::AsmDialect::AD_Intel;
  }
  llvm_unreachable("unknown AsmDialect type");
}

static LLVM_ATTRIBUTE_UNUSED ::llvm::GlobalValue::LinkageTypes convertLinkageToLLVM(::mlir::LLVM::linkage::Linkage value) {
  switch (value) {
  case ::mlir::LLVM::linkage::Linkage::Private:
    return ::llvm::GlobalValue::LinkageTypes::PrivateLinkage;
  case ::mlir::LLVM::linkage::Linkage::Internal:
    return ::llvm::GlobalValue::LinkageTypes::InternalLinkage;
  case ::mlir::LLVM::linkage::Linkage::AvailableExternally:
    return ::llvm::GlobalValue::LinkageTypes::AvailableExternallyLinkage;
  case ::mlir::LLVM::linkage::Linkage::Linkonce:
    return ::llvm::GlobalValue::LinkageTypes::LinkOnceAnyLinkage;
  case ::mlir::LLVM::linkage::Linkage::Weak:
    return ::llvm::GlobalValue::LinkageTypes::WeakAnyLinkage;
  case ::mlir::LLVM::linkage::Linkage::Common:
    return ::llvm::GlobalValue::LinkageTypes::CommonLinkage;
  case ::mlir::LLVM::linkage::Linkage::Appending:
    return ::llvm::GlobalValue::LinkageTypes::AppendingLinkage;
  case ::mlir::LLVM::linkage::Linkage::ExternWeak:
    return ::llvm::GlobalValue::LinkageTypes::ExternalWeakLinkage;
  case ::mlir::LLVM::linkage::Linkage::LinkonceODR:
    return ::llvm::GlobalValue::LinkageTypes::LinkOnceODRLinkage;
  case ::mlir::LLVM::linkage::Linkage::WeakODR:
    return ::llvm::GlobalValue::LinkageTypes::WeakODRLinkage;
  case ::mlir::LLVM::linkage::Linkage::External:
    return ::llvm::GlobalValue::LinkageTypes::ExternalLinkage;
  }
  llvm_unreachable("unknown Linkage type");
}

static LLVM_ATTRIBUTE_UNUSED ::llvm::GlobalValue::UnnamedAddr convertUnnamedAddrToLLVM(::mlir::LLVM::UnnamedAddr value) {
  switch (value) {
  case ::mlir::LLVM::UnnamedAddr::None:
    return ::llvm::GlobalValue::UnnamedAddr::None;
  case ::mlir::LLVM::UnnamedAddr::Local:
    return ::llvm::GlobalValue::UnnamedAddr::Local;
  case ::mlir::LLVM::UnnamedAddr::Global:
    return ::llvm::GlobalValue::UnnamedAddr::Global;
  }
  llvm_unreachable("unknown UnnamedAddr type");
}

