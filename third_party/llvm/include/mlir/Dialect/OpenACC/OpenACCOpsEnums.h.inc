/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace acc {
// DefaultValue Clause
enum class ClauseDefaultValue : uint32_t {
  Present = 0,
  None = 1,
};

::llvm::Optional<ClauseDefaultValue> symbolizeClauseDefaultValue(uint32_t);
::llvm::StringRef stringifyClauseDefaultValue(ClauseDefaultValue);
::llvm::Optional<ClauseDefaultValue> symbolizeClauseDefaultValue(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForClauseDefaultValue() {
  return 1;
}


inline ::llvm::StringRef stringifyEnum(ClauseDefaultValue enumValue) {
  return stringifyClauseDefaultValue(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<ClauseDefaultValue> symbolizeEnum<ClauseDefaultValue>(::llvm::StringRef str) {
  return symbolizeClauseDefaultValue(str);
}
} // namespace acc
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::acc::ClauseDefaultValue> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::acc::ClauseDefaultValue getEmptyKey() {
    return static_cast<::mlir::acc::ClauseDefaultValue>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::acc::ClauseDefaultValue getTombstoneKey() {
    return static_cast<::mlir::acc::ClauseDefaultValue>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::acc::ClauseDefaultValue &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::acc::ClauseDefaultValue &lhs, const ::mlir::acc::ClauseDefaultValue &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace acc {
// built-in reduction operations supported by OpenACC
enum class ReductionOp : uint32_t {
  redop_add = 0,
  redop_mul = 1,
  redop_max = 2,
  redop_min = 3,
  redop_and = 4,
  redop_or = 5,
  redop_xor = 6,
  redop_leqv = 7,
  redop_lneqv = 8,
  redop_land = 9,
  redop_lor = 10,
};

::llvm::Optional<ReductionOp> symbolizeReductionOp(uint32_t);
::llvm::StringRef stringifyReductionOp(ReductionOp);
::llvm::Optional<ReductionOp> symbolizeReductionOp(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForReductionOp() {
  return 10;
}


inline ::llvm::StringRef stringifyEnum(ReductionOp enumValue) {
  return stringifyReductionOp(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<ReductionOp> symbolizeEnum<ReductionOp>(::llvm::StringRef str) {
  return symbolizeReductionOp(str);
}
} // namespace acc
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::acc::ReductionOp> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::acc::ReductionOp getEmptyKey() {
    return static_cast<::mlir::acc::ReductionOp>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::acc::ReductionOp getTombstoneKey() {
    return static_cast<::mlir::acc::ReductionOp>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::acc::ReductionOp &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::acc::ReductionOp &lhs, const ::mlir::acc::ReductionOp &rhs) {
    return lhs == rhs;
  }
};
}

