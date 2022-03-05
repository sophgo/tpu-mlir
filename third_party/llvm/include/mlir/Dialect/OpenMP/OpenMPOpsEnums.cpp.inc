/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Definitions                                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace omp {
::llvm::StringRef stringifyClauseDepend(ClauseDepend val) {
  switch (val) {
    case ClauseDepend::dependsource: return "dependsource";
    case ClauseDepend::dependsink: return "dependsink";
  }
  return "";
}

::llvm::Optional<ClauseDepend> symbolizeClauseDepend(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<ClauseDepend>>(str)
      .Case("dependsource", ClauseDepend::dependsource)
      .Case("dependsink", ClauseDepend::dependsink)
      .Default(::llvm::None);
}
::llvm::Optional<ClauseDepend> symbolizeClauseDepend(uint32_t value) {
  switch (value) {
  case 0: return ClauseDepend::dependsource;
  case 1: return ClauseDepend::dependsink;
  default: return ::llvm::None;
  }
}

} // namespace omp
} // namespace mlir

namespace mlir {
namespace omp {
::llvm::StringRef stringifyClauseMemoryOrderKind(ClauseMemoryOrderKind val) {
  switch (val) {
    case ClauseMemoryOrderKind::seq_cst: return "seq_cst";
    case ClauseMemoryOrderKind::acq_rel: return "acq_rel";
    case ClauseMemoryOrderKind::acquire: return "acquire";
    case ClauseMemoryOrderKind::release: return "release";
    case ClauseMemoryOrderKind::relaxed: return "relaxed";
  }
  return "";
}

::llvm::Optional<ClauseMemoryOrderKind> symbolizeClauseMemoryOrderKind(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<ClauseMemoryOrderKind>>(str)
      .Case("seq_cst", ClauseMemoryOrderKind::seq_cst)
      .Case("acq_rel", ClauseMemoryOrderKind::acq_rel)
      .Case("acquire", ClauseMemoryOrderKind::acquire)
      .Case("release", ClauseMemoryOrderKind::release)
      .Case("relaxed", ClauseMemoryOrderKind::relaxed)
      .Default(::llvm::None);
}
::llvm::Optional<ClauseMemoryOrderKind> symbolizeClauseMemoryOrderKind(uint32_t value) {
  switch (value) {
  case 0: return ClauseMemoryOrderKind::seq_cst;
  case 1: return ClauseMemoryOrderKind::acq_rel;
  case 2: return ClauseMemoryOrderKind::acquire;
  case 3: return ClauseMemoryOrderKind::release;
  case 4: return ClauseMemoryOrderKind::relaxed;
  default: return ::llvm::None;
  }
}

} // namespace omp
} // namespace mlir

namespace mlir {
namespace omp {
::llvm::StringRef stringifyClauseOrderKind(ClauseOrderKind val) {
  switch (val) {
    case ClauseOrderKind::concurrent: return "concurrent";
  }
  return "";
}

::llvm::Optional<ClauseOrderKind> symbolizeClauseOrderKind(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<ClauseOrderKind>>(str)
      .Case("concurrent", ClauseOrderKind::concurrent)
      .Default(::llvm::None);
}
::llvm::Optional<ClauseOrderKind> symbolizeClauseOrderKind(uint32_t value) {
  switch (value) {
  case 1: return ClauseOrderKind::concurrent;
  default: return ::llvm::None;
  }
}

} // namespace omp
} // namespace mlir

namespace mlir {
namespace omp {
::llvm::StringRef stringifyClauseProcBindKind(ClauseProcBindKind val) {
  switch (val) {
    case ClauseProcBindKind::primary: return "primary";
    case ClauseProcBindKind::master: return "master";
    case ClauseProcBindKind::close: return "close";
    case ClauseProcBindKind::spread: return "spread";
  }
  return "";
}

::llvm::Optional<ClauseProcBindKind> symbolizeClauseProcBindKind(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<ClauseProcBindKind>>(str)
      .Case("primary", ClauseProcBindKind::primary)
      .Case("master", ClauseProcBindKind::master)
      .Case("close", ClauseProcBindKind::close)
      .Case("spread", ClauseProcBindKind::spread)
      .Default(::llvm::None);
}
::llvm::Optional<ClauseProcBindKind> symbolizeClauseProcBindKind(uint32_t value) {
  switch (value) {
  case 0: return ClauseProcBindKind::primary;
  case 1: return ClauseProcBindKind::master;
  case 2: return ClauseProcBindKind::close;
  case 3: return ClauseProcBindKind::spread;
  default: return ::llvm::None;
  }
}

} // namespace omp
} // namespace mlir

namespace mlir {
namespace omp {
::llvm::StringRef stringifyClauseScheduleKind(ClauseScheduleKind val) {
  switch (val) {
    case ClauseScheduleKind::Static: return "Static";
    case ClauseScheduleKind::Dynamic: return "Dynamic";
    case ClauseScheduleKind::Guided: return "Guided";
    case ClauseScheduleKind::Auto: return "Auto";
    case ClauseScheduleKind::Runtime: return "Runtime";
  }
  return "";
}

::llvm::Optional<ClauseScheduleKind> symbolizeClauseScheduleKind(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<ClauseScheduleKind>>(str)
      .Case("Static", ClauseScheduleKind::Static)
      .Case("Dynamic", ClauseScheduleKind::Dynamic)
      .Case("Guided", ClauseScheduleKind::Guided)
      .Case("Auto", ClauseScheduleKind::Auto)
      .Case("Runtime", ClauseScheduleKind::Runtime)
      .Default(::llvm::None);
}
::llvm::Optional<ClauseScheduleKind> symbolizeClauseScheduleKind(uint32_t value) {
  switch (value) {
  case 0: return ClauseScheduleKind::Static;
  case 1: return ClauseScheduleKind::Dynamic;
  case 2: return ClauseScheduleKind::Guided;
  case 3: return ClauseScheduleKind::Auto;
  case 4: return ClauseScheduleKind::Runtime;
  default: return ::llvm::None;
  }
}

} // namespace omp
} // namespace mlir

namespace mlir {
namespace omp {
::llvm::StringRef stringifyScheduleModifier(ScheduleModifier val) {
  switch (val) {
    case ScheduleModifier::none: return "none";
    case ScheduleModifier::monotonic: return "monotonic";
    case ScheduleModifier::nonmonotonic: return "nonmonotonic";
    case ScheduleModifier::simd: return "simd";
  }
  return "";
}

::llvm::Optional<ScheduleModifier> symbolizeScheduleModifier(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<ScheduleModifier>>(str)
      .Case("none", ScheduleModifier::none)
      .Case("monotonic", ScheduleModifier::monotonic)
      .Case("nonmonotonic", ScheduleModifier::nonmonotonic)
      .Case("simd", ScheduleModifier::simd)
      .Default(::llvm::None);
}
::llvm::Optional<ScheduleModifier> symbolizeScheduleModifier(uint32_t value) {
  switch (value) {
  case 0: return ScheduleModifier::none;
  case 1: return ScheduleModifier::monotonic;
  case 2: return ScheduleModifier::nonmonotonic;
  case 3: return ScheduleModifier::simd;
  default: return ::llvm::None;
  }
}

} // namespace omp
} // namespace mlir

