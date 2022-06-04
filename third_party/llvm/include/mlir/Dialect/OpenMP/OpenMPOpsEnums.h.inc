/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace omp {
// CancellationConstructType Clause
enum class ClauseCancellationConstructType : uint32_t {
  Parallel = 0,
  Loop = 1,
  Sections = 2,
  Taskgroup = 3,
};

::llvm::Optional<ClauseCancellationConstructType> symbolizeClauseCancellationConstructType(uint32_t);
::llvm::StringRef stringifyClauseCancellationConstructType(ClauseCancellationConstructType);
::llvm::Optional<ClauseCancellationConstructType> symbolizeClauseCancellationConstructType(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForClauseCancellationConstructType() {
  return 3;
}


inline ::llvm::StringRef stringifyEnum(ClauseCancellationConstructType enumValue) {
  return stringifyClauseCancellationConstructType(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<ClauseCancellationConstructType> symbolizeEnum<ClauseCancellationConstructType>(::llvm::StringRef str) {
  return symbolizeClauseCancellationConstructType(str);
}
} // namespace omp
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::omp::ClauseCancellationConstructType> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::omp::ClauseCancellationConstructType getEmptyKey() {
    return static_cast<::mlir::omp::ClauseCancellationConstructType>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::omp::ClauseCancellationConstructType getTombstoneKey() {
    return static_cast<::mlir::omp::ClauseCancellationConstructType>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::omp::ClauseCancellationConstructType &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::omp::ClauseCancellationConstructType &lhs, const ::mlir::omp::ClauseCancellationConstructType &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace omp {
// depend clause
enum class ClauseDepend : uint32_t {
  dependsource = 0,
  dependsink = 1,
};

::llvm::Optional<ClauseDepend> symbolizeClauseDepend(uint32_t);
::llvm::StringRef stringifyClauseDepend(ClauseDepend);
::llvm::Optional<ClauseDepend> symbolizeClauseDepend(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForClauseDepend() {
  return 1;
}


inline ::llvm::StringRef stringifyEnum(ClauseDepend enumValue) {
  return stringifyClauseDepend(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<ClauseDepend> symbolizeEnum<ClauseDepend>(::llvm::StringRef str) {
  return symbolizeClauseDepend(str);
}
} // namespace omp
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::omp::ClauseDepend> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::omp::ClauseDepend getEmptyKey() {
    return static_cast<::mlir::omp::ClauseDepend>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::omp::ClauseDepend getTombstoneKey() {
    return static_cast<::mlir::omp::ClauseDepend>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::omp::ClauseDepend &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::omp::ClauseDepend &lhs, const ::mlir::omp::ClauseDepend &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace omp {
// MemoryOrderKind Clause
enum class ClauseMemoryOrderKind : uint32_t {
  Seq_cst = 0,
  Acq_rel = 1,
  Acquire = 2,
  Release = 3,
  Relaxed = 4,
};

::llvm::Optional<ClauseMemoryOrderKind> symbolizeClauseMemoryOrderKind(uint32_t);
::llvm::StringRef stringifyClauseMemoryOrderKind(ClauseMemoryOrderKind);
::llvm::Optional<ClauseMemoryOrderKind> symbolizeClauseMemoryOrderKind(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForClauseMemoryOrderKind() {
  return 4;
}


inline ::llvm::StringRef stringifyEnum(ClauseMemoryOrderKind enumValue) {
  return stringifyClauseMemoryOrderKind(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<ClauseMemoryOrderKind> symbolizeEnum<ClauseMemoryOrderKind>(::llvm::StringRef str) {
  return symbolizeClauseMemoryOrderKind(str);
}
} // namespace omp
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::omp::ClauseMemoryOrderKind> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::omp::ClauseMemoryOrderKind getEmptyKey() {
    return static_cast<::mlir::omp::ClauseMemoryOrderKind>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::omp::ClauseMemoryOrderKind getTombstoneKey() {
    return static_cast<::mlir::omp::ClauseMemoryOrderKind>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::omp::ClauseMemoryOrderKind &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::omp::ClauseMemoryOrderKind &lhs, const ::mlir::omp::ClauseMemoryOrderKind &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace omp {
// OrderKind Clause
enum class ClauseOrderKind : uint32_t {
  Concurrent = 1,
};

::llvm::Optional<ClauseOrderKind> symbolizeClauseOrderKind(uint32_t);
::llvm::StringRef stringifyClauseOrderKind(ClauseOrderKind);
::llvm::Optional<ClauseOrderKind> symbolizeClauseOrderKind(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForClauseOrderKind() {
  return 1;
}


inline ::llvm::StringRef stringifyEnum(ClauseOrderKind enumValue) {
  return stringifyClauseOrderKind(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<ClauseOrderKind> symbolizeEnum<ClauseOrderKind>(::llvm::StringRef str) {
  return symbolizeClauseOrderKind(str);
}
} // namespace omp
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::omp::ClauseOrderKind> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::omp::ClauseOrderKind getEmptyKey() {
    return static_cast<::mlir::omp::ClauseOrderKind>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::omp::ClauseOrderKind getTombstoneKey() {
    return static_cast<::mlir::omp::ClauseOrderKind>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::omp::ClauseOrderKind &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::omp::ClauseOrderKind &lhs, const ::mlir::omp::ClauseOrderKind &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace omp {
// ProcBindKind Clause
enum class ClauseProcBindKind : uint32_t {
  Primary = 0,
  Master = 1,
  Close = 2,
  Spread = 3,
};

::llvm::Optional<ClauseProcBindKind> symbolizeClauseProcBindKind(uint32_t);
::llvm::StringRef stringifyClauseProcBindKind(ClauseProcBindKind);
::llvm::Optional<ClauseProcBindKind> symbolizeClauseProcBindKind(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForClauseProcBindKind() {
  return 3;
}


inline ::llvm::StringRef stringifyEnum(ClauseProcBindKind enumValue) {
  return stringifyClauseProcBindKind(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<ClauseProcBindKind> symbolizeEnum<ClauseProcBindKind>(::llvm::StringRef str) {
  return symbolizeClauseProcBindKind(str);
}
} // namespace omp
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::omp::ClauseProcBindKind> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::omp::ClauseProcBindKind getEmptyKey() {
    return static_cast<::mlir::omp::ClauseProcBindKind>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::omp::ClauseProcBindKind getTombstoneKey() {
    return static_cast<::mlir::omp::ClauseProcBindKind>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::omp::ClauseProcBindKind &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::omp::ClauseProcBindKind &lhs, const ::mlir::omp::ClauseProcBindKind &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace omp {
// ScheduleKind Clause
enum class ClauseScheduleKind : uint32_t {
  Static = 0,
  Dynamic = 1,
  Guided = 2,
  Auto = 3,
  Runtime = 4,
};

::llvm::Optional<ClauseScheduleKind> symbolizeClauseScheduleKind(uint32_t);
::llvm::StringRef stringifyClauseScheduleKind(ClauseScheduleKind);
::llvm::Optional<ClauseScheduleKind> symbolizeClauseScheduleKind(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForClauseScheduleKind() {
  return 4;
}


inline ::llvm::StringRef stringifyEnum(ClauseScheduleKind enumValue) {
  return stringifyClauseScheduleKind(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<ClauseScheduleKind> symbolizeEnum<ClauseScheduleKind>(::llvm::StringRef str) {
  return symbolizeClauseScheduleKind(str);
}
} // namespace omp
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::omp::ClauseScheduleKind> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::omp::ClauseScheduleKind getEmptyKey() {
    return static_cast<::mlir::omp::ClauseScheduleKind>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::omp::ClauseScheduleKind getTombstoneKey() {
    return static_cast<::mlir::omp::ClauseScheduleKind>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::omp::ClauseScheduleKind &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::omp::ClauseScheduleKind &lhs, const ::mlir::omp::ClauseScheduleKind &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace omp {
// OpenMP Schedule Modifier
enum class ScheduleModifier : uint32_t {
  none = 0,
  monotonic = 1,
  nonmonotonic = 2,
  simd = 3,
};

::llvm::Optional<ScheduleModifier> symbolizeScheduleModifier(uint32_t);
::llvm::StringRef stringifyScheduleModifier(ScheduleModifier);
::llvm::Optional<ScheduleModifier> symbolizeScheduleModifier(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForScheduleModifier() {
  return 3;
}


inline ::llvm::StringRef stringifyEnum(ScheduleModifier enumValue) {
  return stringifyScheduleModifier(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<ScheduleModifier> symbolizeEnum<ScheduleModifier>(::llvm::StringRef str) {
  return symbolizeScheduleModifier(str);
}
} // namespace omp
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::omp::ScheduleModifier> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::omp::ScheduleModifier getEmptyKey() {
    return static_cast<::mlir::omp::ScheduleModifier>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::omp::ScheduleModifier getTombstoneKey() {
    return static_cast<::mlir::omp::ScheduleModifier>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::omp::ScheduleModifier &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::omp::ScheduleModifier &lhs, const ::mlir::omp::ScheduleModifier &rhs) {
    return lhs == rhs;
  }
};
}

