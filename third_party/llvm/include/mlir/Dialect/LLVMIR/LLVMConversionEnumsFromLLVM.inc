inline LLVM_ATTRIBUTE_UNUSED ::mlir::LLVM::AsmDialect convertAsmDialectFromLLVM(::llvm::InlineAsm::AsmDialect value) {
  switch (value) {
  case ::llvm::InlineAsm::AsmDialect::AD_ATT:
    return ::mlir::LLVM::AsmDialect::AD_ATT;
  case ::llvm::InlineAsm::AsmDialect::AD_Intel:
    return ::mlir::LLVM::AsmDialect::AD_Intel;
  }
  llvm_unreachable("unknown ::llvm::InlineAsm::AsmDialect type");}

inline LLVM_ATTRIBUTE_UNUSED ::mlir::LLVM::linkage::Linkage convertLinkageFromLLVM(::llvm::GlobalValue::LinkageTypes value) {
  switch (value) {
  case ::llvm::GlobalValue::LinkageTypes::PrivateLinkage:
    return ::mlir::LLVM::linkage::Linkage::Private;
  case ::llvm::GlobalValue::LinkageTypes::InternalLinkage:
    return ::mlir::LLVM::linkage::Linkage::Internal;
  case ::llvm::GlobalValue::LinkageTypes::AvailableExternallyLinkage:
    return ::mlir::LLVM::linkage::Linkage::AvailableExternally;
  case ::llvm::GlobalValue::LinkageTypes::LinkOnceAnyLinkage:
    return ::mlir::LLVM::linkage::Linkage::Linkonce;
  case ::llvm::GlobalValue::LinkageTypes::WeakAnyLinkage:
    return ::mlir::LLVM::linkage::Linkage::Weak;
  case ::llvm::GlobalValue::LinkageTypes::CommonLinkage:
    return ::mlir::LLVM::linkage::Linkage::Common;
  case ::llvm::GlobalValue::LinkageTypes::AppendingLinkage:
    return ::mlir::LLVM::linkage::Linkage::Appending;
  case ::llvm::GlobalValue::LinkageTypes::ExternalWeakLinkage:
    return ::mlir::LLVM::linkage::Linkage::ExternWeak;
  case ::llvm::GlobalValue::LinkageTypes::LinkOnceODRLinkage:
    return ::mlir::LLVM::linkage::Linkage::LinkonceODR;
  case ::llvm::GlobalValue::LinkageTypes::WeakODRLinkage:
    return ::mlir::LLVM::linkage::Linkage::WeakODR;
  case ::llvm::GlobalValue::LinkageTypes::ExternalLinkage:
    return ::mlir::LLVM::linkage::Linkage::External;
  }
  llvm_unreachable("unknown ::llvm::GlobalValue::LinkageTypes type");}

inline LLVM_ATTRIBUTE_UNUSED ::mlir::LLVM::UnnamedAddr convertUnnamedAddrFromLLVM(::llvm::GlobalValue::UnnamedAddr value) {
  switch (value) {
  case ::llvm::GlobalValue::UnnamedAddr::None:
    return ::mlir::LLVM::UnnamedAddr::None;
  case ::llvm::GlobalValue::UnnamedAddr::Local:
    return ::mlir::LLVM::UnnamedAddr::Local;
  case ::llvm::GlobalValue::UnnamedAddr::Global:
    return ::mlir::LLVM::UnnamedAddr::Global;
  }
  llvm_unreachable("unknown ::llvm::GlobalValue::UnnamedAddr type");}

