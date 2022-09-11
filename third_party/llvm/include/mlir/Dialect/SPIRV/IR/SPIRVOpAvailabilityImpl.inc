/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* SPIR-V Op Availability Implementations                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> AccessChainOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> AccessChainOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> AccessChainOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> AccessChainOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> AddressOfOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> AddressOfOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> AddressOfOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> AddressOfOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> AtomicAndOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> AtomicAndOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> AtomicAndOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> AtomicAndOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> AtomicCompareExchangeOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->equal_semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->unequal_semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> AtomicCompareExchangeOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->equal_semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->unequal_semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> AtomicCompareExchangeOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> AtomicCompareExchangeOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->equal_semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->unequal_semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> AtomicCompareExchangeWeakOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Kernel}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->equal_semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->unequal_semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> AtomicCompareExchangeWeakOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->equal_semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->unequal_semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> AtomicCompareExchangeWeakOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_3)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_3; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> AtomicCompareExchangeWeakOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->equal_semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->unequal_semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> AtomicExchangeOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> AtomicExchangeOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> AtomicExchangeOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> AtomicExchangeOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> AtomicIAddOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> AtomicIAddOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> AtomicIAddOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> AtomicIAddOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> AtomicIDecrementOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> AtomicIDecrementOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> AtomicIDecrementOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> AtomicIDecrementOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> AtomicIIncrementOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> AtomicIIncrementOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> AtomicIIncrementOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> AtomicIIncrementOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> AtomicISubOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> AtomicISubOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> AtomicISubOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> AtomicISubOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> AtomicOrOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> AtomicOrOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> AtomicOrOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> AtomicOrOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> AtomicSMaxOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> AtomicSMaxOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> AtomicSMaxOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> AtomicSMaxOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> AtomicSMinOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> AtomicSMinOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> AtomicSMinOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> AtomicSMinOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> AtomicUMaxOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> AtomicUMaxOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> AtomicUMaxOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> AtomicUMaxOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> AtomicUMinOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> AtomicUMinOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> AtomicUMinOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> AtomicUMinOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> AtomicXorOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> AtomicXorOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> AtomicXorOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> AtomicXorOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> BitCountOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> BitCountOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> BitCountOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> BitCountOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> BitFieldInsertOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::BitInstructions, ::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> BitFieldInsertOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> BitFieldInsertOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> BitFieldInsertOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> BitFieldSExtractOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::BitInstructions, ::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> BitFieldSExtractOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> BitFieldSExtractOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> BitFieldSExtractOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> BitFieldUExtractOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::BitInstructions, ::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> BitFieldUExtractOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> BitFieldUExtractOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> BitFieldUExtractOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> BitReverseOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::BitInstructions, ::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> BitReverseOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> BitReverseOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> BitReverseOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> BitcastOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> BitcastOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> BitcastOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> BitcastOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> BitwiseAndOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> BitwiseAndOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> BitwiseAndOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> BitwiseAndOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> BitwiseOrOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> BitwiseOrOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> BitwiseOrOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> BitwiseOrOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> BitwiseXorOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> BitwiseXorOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> BitwiseXorOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> BitwiseXorOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> BranchConditionalOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> BranchConditionalOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> BranchConditionalOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> BranchConditionalOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> BranchOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> BranchOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> BranchOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> BranchOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> CLCeilOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Kernel}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> CLCeilOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> CLCeilOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> CLCeilOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> CLCosOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Kernel}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> CLCosOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> CLCosOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> CLCosOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> CLErfOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Kernel}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> CLErfOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> CLErfOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> CLErfOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> CLExpOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Kernel}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> CLExpOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> CLExpOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> CLExpOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> CLFAbsOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Kernel}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> CLFAbsOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> CLFAbsOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> CLFAbsOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> CLFloorOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Kernel}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> CLFloorOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> CLFloorOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> CLFloorOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> CLFmaOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Kernel}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> CLFmaOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> CLFmaOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> CLFmaOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> CLLogOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Kernel}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> CLLogOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> CLLogOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> CLLogOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> CLPowOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Kernel}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> CLPowOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> CLPowOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> CLPowOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> CLRoundOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Kernel}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> CLRoundOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> CLRoundOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> CLRoundOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> CLRsqrtOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Kernel}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> CLRsqrtOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> CLRsqrtOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> CLRsqrtOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> CLSAbsOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Kernel}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> CLSAbsOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> CLSAbsOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> CLSAbsOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> CLSinOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Kernel}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> CLSinOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> CLSinOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> CLSinOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> CLSqrtOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Kernel}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> CLSqrtOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> CLSqrtOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> CLSqrtOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> CLTanhOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Kernel}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> CLTanhOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> CLTanhOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> CLTanhOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> CompositeConstructOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> CompositeConstructOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> CompositeConstructOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> CompositeConstructOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> CompositeExtractOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> CompositeExtractOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> CompositeExtractOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> CompositeExtractOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> CompositeInsertOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> CompositeInsertOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> CompositeInsertOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> CompositeInsertOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> ConstantOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> ConstantOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> ConstantOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> ConstantOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> ControlBarrierOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->memory_semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> ControlBarrierOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->memory_semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> ControlBarrierOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> ControlBarrierOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->memory_semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> ConvertFToSOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> ConvertFToSOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> ConvertFToSOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> ConvertFToSOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> ConvertFToUOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> ConvertFToUOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> ConvertFToUOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> ConvertFToUOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> ConvertSToFOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> ConvertSToFOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> ConvertSToFOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> ConvertSToFOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> ConvertUToFOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> ConvertUToFOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> ConvertUToFOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> ConvertUToFOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> CopyMemoryOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> CopyMemoryOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> CopyMemoryOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> CopyMemoryOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> EXTAtomicFAddOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::AtomicFloat16AddEXT, ::mlir::spirv::Capability::AtomicFloat32AddEXT, ::mlir::spirv::Capability::AtomicFloat64AddEXT}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> EXTAtomicFAddOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> EXTAtomicFAddOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> EXTAtomicFAddOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_EXT_shader_atomic_float_add}; ArrayRef<::mlir::spirv::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> EntryPointOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->execution_model();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> EntryPointOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> EntryPointOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> EntryPointOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> ExecutionModeOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->execution_mode();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> ExecutionModeOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  {
    auto tblgen_attrVal = this->execution_mode();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> ExecutionModeOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> ExecutionModeOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->execution_mode();
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> FAddOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> FAddOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> FAddOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> FAddOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> FConvertOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> FConvertOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> FConvertOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> FConvertOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> FDivOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> FDivOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> FDivOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> FDivOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> FModOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> FModOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> FModOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> FModOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> FMulOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> FMulOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> FMulOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> FMulOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> FNegateOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> FNegateOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> FNegateOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> FNegateOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> FOrdEqualOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> FOrdEqualOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> FOrdEqualOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> FOrdEqualOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> FOrdGreaterThanEqualOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> FOrdGreaterThanEqualOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> FOrdGreaterThanEqualOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> FOrdGreaterThanEqualOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> FOrdGreaterThanOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> FOrdGreaterThanOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> FOrdGreaterThanOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> FOrdGreaterThanOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> FOrdLessThanEqualOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> FOrdLessThanEqualOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> FOrdLessThanEqualOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> FOrdLessThanEqualOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> FOrdLessThanOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> FOrdLessThanOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> FOrdLessThanOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> FOrdLessThanOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> FOrdNotEqualOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> FOrdNotEqualOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> FOrdNotEqualOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> FOrdNotEqualOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> FRemOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> FRemOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> FRemOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> FRemOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> FSubOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> FSubOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> FSubOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> FSubOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> FUnordEqualOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> FUnordEqualOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> FUnordEqualOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> FUnordEqualOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> FUnordGreaterThanEqualOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> FUnordGreaterThanEqualOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> FUnordGreaterThanEqualOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> FUnordGreaterThanEqualOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> FUnordGreaterThanOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> FUnordGreaterThanOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> FUnordGreaterThanOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> FUnordGreaterThanOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> FUnordLessThanEqualOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> FUnordLessThanEqualOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> FUnordLessThanEqualOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> FUnordLessThanEqualOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> FUnordLessThanOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> FUnordLessThanOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> FUnordLessThanOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> FUnordLessThanOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> FUnordNotEqualOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> FUnordNotEqualOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> FUnordNotEqualOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> FUnordNotEqualOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> FuncOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::FunctionControl tblgen_attrVal = this->function_control() & static_cast<::mlir::spirv::FunctionControl>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> FuncOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> FuncOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> FuncOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> FunctionCallOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> FunctionCallOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> FunctionCallOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> FunctionCallOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLAcosOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLAcosOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLAcosOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLAcosOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLAsinOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLAsinOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLAsinOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLAsinOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLAtanOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLAtanOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLAtanOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLAtanOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLCeilOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLCeilOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLCeilOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLCeilOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLCosOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLCosOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLCosOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLCosOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLCoshOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLCoshOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLCoshOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLCoshOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLExpOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLExpOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLExpOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLExpOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLFAbsOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLFAbsOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLFAbsOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLFAbsOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLFClampOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLFClampOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLFClampOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLFClampOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLFMaxOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLFMaxOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLFMaxOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLFMaxOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLFMinOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLFMinOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLFMinOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLFMinOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLFMixOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLFMixOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLFMixOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLFMixOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLFSignOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLFSignOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLFSignOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLFSignOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLFindUMsbOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLFindUMsbOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLFindUMsbOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLFindUMsbOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLFloorOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLFloorOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLFloorOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLFloorOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLFmaOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLFmaOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLFmaOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLFmaOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLFrexpStructOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLFrexpStructOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLFrexpStructOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLFrexpStructOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLInverseSqrtOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLInverseSqrtOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLInverseSqrtOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLInverseSqrtOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLLdexpOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLLdexpOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLLdexpOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLLdexpOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLLogOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLLogOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLLogOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLLogOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLPowOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLPowOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLPowOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLPowOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLRoundOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLRoundOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLRoundOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLRoundOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLSAbsOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLSAbsOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLSAbsOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLSAbsOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLSClampOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLSClampOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLSClampOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLSClampOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLSMaxOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLSMaxOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLSMaxOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLSMaxOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLSMinOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLSMinOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLSMinOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLSMinOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLSSignOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLSSignOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLSSignOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLSSignOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLSinOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLSinOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLSinOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLSinOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLSinhOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLSinhOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLSinhOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLSinhOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLSqrtOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLSqrtOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLSqrtOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLSqrtOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLTanOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLTanOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLTanOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLTanOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLTanhOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLTanhOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLTanhOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLTanhOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLUClampOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLUClampOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLUClampOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLUClampOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLUMaxOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLUMaxOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLUMaxOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLUMaxOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLUMinOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLUMinOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GLUMinOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLUMinOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GlobalVariableOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GlobalVariableOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GlobalVariableOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GlobalVariableOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GroupBroadcastOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Groups}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GroupBroadcastOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GroupBroadcastOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GroupBroadcastOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GroupNonUniformBallotOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::GroupNonUniformBallot}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GroupNonUniformBallotOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_3)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_3; }};
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GroupNonUniformBallotOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GroupNonUniformBallotOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GroupNonUniformBroadcastOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::GroupNonUniformBallot}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GroupNonUniformBroadcastOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_3)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_3; }};
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GroupNonUniformBroadcastOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GroupNonUniformBroadcastOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GroupNonUniformElectOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::GroupNonUniform}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GroupNonUniformElectOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_3)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_3; }};
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GroupNonUniformElectOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GroupNonUniformElectOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GroupNonUniformFAddOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::GroupNonUniformArithmetic, ::mlir::spirv::Capability::GroupNonUniformClustered, ::mlir::spirv::Capability::GroupNonUniformPartitionedNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GroupNonUniformFAddOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_3)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_3; }};
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GroupNonUniformFAddOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GroupNonUniformFAddOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GroupNonUniformFMaxOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::GroupNonUniformArithmetic, ::mlir::spirv::Capability::GroupNonUniformClustered, ::mlir::spirv::Capability::GroupNonUniformPartitionedNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GroupNonUniformFMaxOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_3)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_3; }};
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GroupNonUniformFMaxOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GroupNonUniformFMaxOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GroupNonUniformFMinOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::GroupNonUniformArithmetic, ::mlir::spirv::Capability::GroupNonUniformClustered, ::mlir::spirv::Capability::GroupNonUniformPartitionedNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GroupNonUniformFMinOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_3)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_3; }};
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GroupNonUniformFMinOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GroupNonUniformFMinOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GroupNonUniformFMulOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::GroupNonUniformArithmetic, ::mlir::spirv::Capability::GroupNonUniformClustered, ::mlir::spirv::Capability::GroupNonUniformPartitionedNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GroupNonUniformFMulOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_3)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_3; }};
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GroupNonUniformFMulOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GroupNonUniformFMulOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GroupNonUniformIAddOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::GroupNonUniformArithmetic, ::mlir::spirv::Capability::GroupNonUniformClustered, ::mlir::spirv::Capability::GroupNonUniformPartitionedNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GroupNonUniformIAddOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_3)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_3; }};
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GroupNonUniformIAddOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GroupNonUniformIAddOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GroupNonUniformIMulOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::GroupNonUniformArithmetic, ::mlir::spirv::Capability::GroupNonUniformClustered, ::mlir::spirv::Capability::GroupNonUniformPartitionedNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GroupNonUniformIMulOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_3)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_3; }};
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GroupNonUniformIMulOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GroupNonUniformIMulOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GroupNonUniformSMaxOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::GroupNonUniformArithmetic, ::mlir::spirv::Capability::GroupNonUniformClustered, ::mlir::spirv::Capability::GroupNonUniformPartitionedNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GroupNonUniformSMaxOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_3)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_3; }};
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GroupNonUniformSMaxOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GroupNonUniformSMaxOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GroupNonUniformSMinOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::GroupNonUniformArithmetic, ::mlir::spirv::Capability::GroupNonUniformClustered, ::mlir::spirv::Capability::GroupNonUniformPartitionedNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GroupNonUniformSMinOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_3)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_3; }};
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GroupNonUniformSMinOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GroupNonUniformSMinOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GroupNonUniformShuffleDownOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::GroupNonUniformShuffleRelative}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GroupNonUniformShuffleDownOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_3)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_3; }};
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GroupNonUniformShuffleDownOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GroupNonUniformShuffleDownOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GroupNonUniformShuffleOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::GroupNonUniformShuffle}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GroupNonUniformShuffleOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_3)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_3; }};
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GroupNonUniformShuffleOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GroupNonUniformShuffleOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GroupNonUniformShuffleUpOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::GroupNonUniformShuffleRelative}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GroupNonUniformShuffleUpOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_3)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_3; }};
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GroupNonUniformShuffleUpOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GroupNonUniformShuffleUpOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GroupNonUniformShuffleXorOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::GroupNonUniformShuffle}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GroupNonUniformShuffleXorOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_3)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_3; }};
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GroupNonUniformShuffleXorOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GroupNonUniformShuffleXorOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GroupNonUniformUMaxOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::GroupNonUniformArithmetic, ::mlir::spirv::Capability::GroupNonUniformClustered, ::mlir::spirv::Capability::GroupNonUniformPartitionedNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GroupNonUniformUMaxOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_3)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_3; }};
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GroupNonUniformUMaxOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GroupNonUniformUMaxOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GroupNonUniformUMinOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::GroupNonUniformArithmetic, ::mlir::spirv::Capability::GroupNonUniformClustered, ::mlir::spirv::Capability::GroupNonUniformPartitionedNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GroupNonUniformUMinOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_3)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_3; }};
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> GroupNonUniformUMinOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GroupNonUniformUMinOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> IAddCarryOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> IAddCarryOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> IAddCarryOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> IAddCarryOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> IAddOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> IAddOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> IAddOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> IAddOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> IEqualOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> IEqualOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> IEqualOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> IEqualOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> IMulOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> IMulOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> IMulOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> IMulOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> INTELJointMatrixLoadOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::JointMatrixINTEL}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  {
    auto tblgen_attrVal = this->scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> INTELJointMatrixLoadOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  {
    auto tblgen_attrVal = this->scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> INTELJointMatrixLoadOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> INTELJointMatrixLoadOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_INTEL_joint_matrix}; ArrayRef<::mlir::spirv::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> INTELJointMatrixMadOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::JointMatrixINTEL}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  {
    auto tblgen_attrVal = this->scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> INTELJointMatrixMadOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  {
    auto tblgen_attrVal = this->scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> INTELJointMatrixMadOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> INTELJointMatrixMadOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_INTEL_joint_matrix}; ArrayRef<::mlir::spirv::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> INTELJointMatrixStoreOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::JointMatrixINTEL}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  {
    auto tblgen_attrVal = this->scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> INTELJointMatrixStoreOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  {
    auto tblgen_attrVal = this->scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> INTELJointMatrixStoreOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> INTELJointMatrixStoreOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_INTEL_joint_matrix}; ArrayRef<::mlir::spirv::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> INTELJointMatrixWorkItemLengthOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::JointMatrixINTEL}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> INTELJointMatrixWorkItemLengthOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> INTELJointMatrixWorkItemLengthOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> INTELJointMatrixWorkItemLengthOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_INTEL_joint_matrix}; ArrayRef<::mlir::spirv::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> INTELSubgroupBlockReadOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::SubgroupBufferBlockIOINTEL}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> INTELSubgroupBlockReadOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> INTELSubgroupBlockReadOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> INTELSubgroupBlockReadOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_INTEL_subgroups}; ArrayRef<::mlir::spirv::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> INTELSubgroupBlockWriteOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::SubgroupBufferBlockIOINTEL}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> INTELSubgroupBlockWriteOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> INTELSubgroupBlockWriteOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> INTELSubgroupBlockWriteOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_INTEL_subgroups}; ArrayRef<::mlir::spirv::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> INotEqualOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> INotEqualOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> INotEqualOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> INotEqualOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> ISubBorrowOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> ISubBorrowOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> ISubBorrowOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> ISubBorrowOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> ISubOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> ISubOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> ISubOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> ISubOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> ImageDrefGatherOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> ImageDrefGatherOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> ImageDrefGatherOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> ImageDrefGatherOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> ImageOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> ImageOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> ImageOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> ImageOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> ImageQuerySizeOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::ImageQuery, ::mlir::spirv::Capability::Kernel}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> ImageQuerySizeOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> ImageQuerySizeOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> ImageQuerySizeOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> InBoundsPtrAccessChainOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Addresses}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> InBoundsPtrAccessChainOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> InBoundsPtrAccessChainOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> InBoundsPtrAccessChainOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> IsInfOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> IsInfOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> IsInfOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> IsInfOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> IsNanOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> IsNanOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> IsNanOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> IsNanOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> KHRAssumeTrueOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::ExpectAssumeKHR}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> KHRAssumeTrueOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> KHRAssumeTrueOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> KHRAssumeTrueOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_KHR_expect_assume}; ArrayRef<::mlir::spirv::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> KHRSubgroupBallotOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::SubgroupBallotKHR}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> KHRSubgroupBallotOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> KHRSubgroupBallotOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> KHRSubgroupBallotOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_KHR_shader_ballot}; ArrayRef<::mlir::spirv::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> LoadOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> LoadOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> LoadOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> LoadOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> LogicalAndOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> LogicalAndOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> LogicalAndOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> LogicalAndOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> LogicalEqualOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> LogicalEqualOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> LogicalEqualOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> LogicalEqualOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> LogicalNotEqualOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> LogicalNotEqualOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> LogicalNotEqualOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> LogicalNotEqualOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> LogicalNotOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> LogicalNotOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> LogicalNotOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> LogicalNotOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> LogicalOrOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> LogicalOrOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> LogicalOrOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> LogicalOrOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> LoopOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::LoopControl tblgen_attrVal = this->loop_control() & static_cast<::mlir::spirv::LoopControl>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> LoopOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::LoopControl tblgen_attrVal = this->loop_control() & static_cast<::mlir::spirv::LoopControl>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> LoopOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> LoopOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::LoopControl tblgen_attrVal = this->loop_control() & static_cast<::mlir::spirv::LoopControl>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> MatrixTimesMatrixOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Matrix}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> MatrixTimesMatrixOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> MatrixTimesMatrixOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> MatrixTimesMatrixOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> MatrixTimesScalarOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Matrix}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> MatrixTimesScalarOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> MatrixTimesScalarOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> MatrixTimesScalarOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> MemoryBarrierOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->memory_semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> MemoryBarrierOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->memory_semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, *tblgen_instance)); } else { tblgen_overall = *tblgen_instance; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> MemoryBarrierOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> MemoryBarrierOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->memory_semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> MergeOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> MergeOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> MergeOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> MergeOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> ModuleOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->addressing_model();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  {
    auto tblgen_attrVal = this->memory_model();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> ModuleOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> ModuleOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> ModuleOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->addressing_model();
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  {
    auto tblgen_attrVal = this->memory_model();
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> NVCooperativeMatrixLengthOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::CooperativeMatrixNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> NVCooperativeMatrixLengthOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> NVCooperativeMatrixLengthOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> NVCooperativeMatrixLengthOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_cooperative_matrix}; ArrayRef<::mlir::spirv::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> NVCooperativeMatrixLoadOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::CooperativeMatrixNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> NVCooperativeMatrixLoadOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> NVCooperativeMatrixLoadOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> NVCooperativeMatrixLoadOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_cooperative_matrix}; ArrayRef<::mlir::spirv::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> NVCooperativeMatrixMulAddOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::CooperativeMatrixNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> NVCooperativeMatrixMulAddOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> NVCooperativeMatrixMulAddOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> NVCooperativeMatrixMulAddOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_cooperative_matrix}; ArrayRef<::mlir::spirv::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> NVCooperativeMatrixStoreOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::CooperativeMatrixNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> NVCooperativeMatrixStoreOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> NVCooperativeMatrixStoreOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> NVCooperativeMatrixStoreOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_cooperative_matrix}; ArrayRef<::mlir::spirv::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> NotOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> NotOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> NotOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> NotOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> OrderedOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Kernel}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> OrderedOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> OrderedOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> OrderedOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> PtrAccessChainOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Addresses, ::mlir::spirv::Capability::PhysicalStorageBufferAddresses, ::mlir::spirv::Capability::VariablePointers, ::mlir::spirv::Capability::VariablePointersStorageBuffer}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> PtrAccessChainOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> PtrAccessChainOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> PtrAccessChainOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> ReferenceOfOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> ReferenceOfOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> ReferenceOfOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> ReferenceOfOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> ReturnOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> ReturnOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> ReturnOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> ReturnOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> ReturnValueOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> ReturnValueOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> ReturnValueOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> ReturnValueOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> SConvertOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> SConvertOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> SConvertOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> SConvertOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> SDivOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> SDivOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> SDivOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> SDivOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> SGreaterThanEqualOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> SGreaterThanEqualOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> SGreaterThanEqualOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> SGreaterThanEqualOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> SGreaterThanOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> SGreaterThanOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> SGreaterThanOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> SGreaterThanOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> SLessThanEqualOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> SLessThanEqualOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> SLessThanEqualOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> SLessThanEqualOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> SLessThanOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> SLessThanOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> SLessThanOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> SLessThanOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> SModOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> SModOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> SModOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> SModOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> SNegateOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> SNegateOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> SNegateOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> SNegateOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> SRemOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> SRemOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> SRemOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> SRemOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> SelectOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> SelectOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> SelectOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> SelectOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> SelectionOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> SelectionOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> SelectionOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> SelectionOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> ShiftLeftLogicalOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> ShiftLeftLogicalOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> ShiftLeftLogicalOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> ShiftLeftLogicalOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> ShiftRightArithmeticOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> ShiftRightArithmeticOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> ShiftRightArithmeticOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> ShiftRightArithmeticOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> ShiftRightLogicalOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> ShiftRightLogicalOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> ShiftRightLogicalOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> ShiftRightLogicalOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> SpecConstantCompositeOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> SpecConstantCompositeOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> SpecConstantCompositeOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> SpecConstantCompositeOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> SpecConstantOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> SpecConstantOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> SpecConstantOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> SpecConstantOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> SpecConstantOperationOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> SpecConstantOperationOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> SpecConstantOperationOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> SpecConstantOperationOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> StoreOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> StoreOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> StoreOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> StoreOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> TransposeOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Matrix}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> TransposeOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> TransposeOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> TransposeOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> UConvertOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> UConvertOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> UConvertOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> UConvertOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> UDivOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> UDivOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> UDivOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> UDivOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> UGreaterThanEqualOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> UGreaterThanEqualOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> UGreaterThanEqualOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> UGreaterThanEqualOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> UGreaterThanOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> UGreaterThanOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> UGreaterThanOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> UGreaterThanOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> ULessThanEqualOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> ULessThanEqualOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> ULessThanEqualOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> ULessThanEqualOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> ULessThanOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> ULessThanOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> ULessThanOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> ULessThanOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> UModOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> UModOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> UModOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> UModOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> UndefOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> UndefOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> UndefOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> UndefOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> UnorderedOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Kernel}; ArrayRef<::mlir::spirv::Capability> ref(caps, std::size(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> UnorderedOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> UnorderedOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> UnorderedOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> UnreachableOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> UnreachableOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> UnreachableOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> UnreachableOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> VariableOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->storage_class();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> VariableOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> VariableOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> VariableOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->storage_class();
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> VectorExtractDynamicOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> VectorExtractDynamicOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> VectorExtractDynamicOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> VectorExtractDynamicOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> VectorInsertDynamicOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> VectorInsertDynamicOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> VectorInsertDynamicOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> VectorInsertDynamicOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> VectorShuffleOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> VectorShuffleOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> VectorShuffleOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> VectorShuffleOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> VectorTimesScalarOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> VectorTimesScalarOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> VectorTimesScalarOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> VectorTimesScalarOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> YieldOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> YieldOp::getMinVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(*tblgen_overall, ::mlir::spirv::Version::V_1_0)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_0; }};
  }
  return tblgen_overall;
}
llvm::Optional<::mlir::spirv::Version> YieldOp::getMaxVersion() {
  llvm::Optional<::mlir::spirv::Version> tblgen_overall = ::llvm::None;
  {
    
    { if (tblgen_overall) { tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(*tblgen_overall, ::mlir::spirv::Version::V_1_6)); } else { tblgen_overall = ::mlir::spirv::Version::V_1_6; }};
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> YieldOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
