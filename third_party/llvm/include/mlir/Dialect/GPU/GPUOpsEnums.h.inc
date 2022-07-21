/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace gpu {
// built-in reduction operations supported by gpu.allreduce.
enum class AllReduceOperation : uint32_t {
  ADD = 0,
  AND = 1,
  MAX = 2,
  MIN = 3,
  MUL = 4,
  OR = 5,
  XOR = 6,
};

::llvm::Optional<AllReduceOperation> symbolizeAllReduceOperation(uint32_t);
::llvm::StringRef stringifyAllReduceOperation(AllReduceOperation);
::llvm::Optional<AllReduceOperation> symbolizeAllReduceOperation(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForAllReduceOperation() {
  return 6;
}


inline ::llvm::StringRef stringifyEnum(AllReduceOperation enumValue) {
  return stringifyAllReduceOperation(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<AllReduceOperation> symbolizeEnum<AllReduceOperation>(::llvm::StringRef str) {
  return symbolizeAllReduceOperation(str);
}
} // namespace gpu
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::gpu::AllReduceOperation> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::gpu::AllReduceOperation getEmptyKey() {
    return static_cast<::mlir::gpu::AllReduceOperation>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::gpu::AllReduceOperation getTombstoneKey() {
    return static_cast<::mlir::gpu::AllReduceOperation>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::gpu::AllReduceOperation &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::gpu::AllReduceOperation &lhs, const ::mlir::gpu::AllReduceOperation &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace gpu {
// a dimension, either 'x', 'y', or 'z'
enum class Dimension : uint32_t {
  x = 0,
  y = 1,
  z = 2,
};

::llvm::Optional<Dimension> symbolizeDimension(uint32_t);
::llvm::StringRef stringifyDimension(Dimension);
::llvm::Optional<Dimension> symbolizeDimension(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForDimension() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(Dimension enumValue) {
  return stringifyDimension(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<Dimension> symbolizeEnum<Dimension>(::llvm::StringRef str) {
  return symbolizeDimension(str);
}
} // namespace gpu
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::gpu::Dimension> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::gpu::Dimension getEmptyKey() {
    return static_cast<::mlir::gpu::Dimension>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::gpu::Dimension getTombstoneKey() {
    return static_cast<::mlir::gpu::Dimension>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::gpu::Dimension &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::gpu::Dimension &lhs, const ::mlir::gpu::Dimension &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace gpu {
// Indexing modes supported by gpu.shuffle.
enum class ShuffleMode : uint32_t {
  XOR = 0,
  UP = 2,
  DOWN = 1,
  IDX = 3,
};

::llvm::Optional<ShuffleMode> symbolizeShuffleMode(uint32_t);
::llvm::StringRef stringifyShuffleMode(ShuffleMode);
::llvm::Optional<ShuffleMode> symbolizeShuffleMode(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForShuffleMode() {
  return 3;
}


inline ::llvm::StringRef stringifyEnum(ShuffleMode enumValue) {
  return stringifyShuffleMode(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<ShuffleMode> symbolizeEnum<ShuffleMode>(::llvm::StringRef str) {
  return symbolizeShuffleMode(str);
}
} // namespace gpu
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::gpu::ShuffleMode> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::gpu::ShuffleMode getEmptyKey() {
    return static_cast<::mlir::gpu::ShuffleMode>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::gpu::ShuffleMode getTombstoneKey() {
    return static_cast<::mlir::gpu::ShuffleMode>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::gpu::ShuffleMode &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::gpu::ShuffleMode &lhs, const ::mlir::gpu::ShuffleMode &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace gpu {
// elementwise operation to apply to mma matrix
enum class MMAElementwiseOp : uint32_t {
  ADDF = 0,
  MULF = 1,
  MAXF = 2,
  MINF = 3,
  DIVF = 4,
};

::llvm::Optional<MMAElementwiseOp> symbolizeMMAElementwiseOp(uint32_t);
::llvm::StringRef stringifyMMAElementwiseOp(MMAElementwiseOp);
::llvm::Optional<MMAElementwiseOp> symbolizeMMAElementwiseOp(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForMMAElementwiseOp() {
  return 4;
}


inline ::llvm::StringRef stringifyEnum(MMAElementwiseOp enumValue) {
  return stringifyMMAElementwiseOp(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<MMAElementwiseOp> symbolizeEnum<MMAElementwiseOp>(::llvm::StringRef str) {
  return symbolizeMMAElementwiseOp(str);
}
} // namespace gpu
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::gpu::MMAElementwiseOp> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::gpu::MMAElementwiseOp getEmptyKey() {
    return static_cast<::mlir::gpu::MMAElementwiseOp>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::gpu::MMAElementwiseOp getTombstoneKey() {
    return static_cast<::mlir::gpu::MMAElementwiseOp>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::gpu::MMAElementwiseOp &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::gpu::MMAElementwiseOp &lhs, const ::mlir::gpu::MMAElementwiseOp &rhs) {
    return lhs == rhs;
  }
};
}

