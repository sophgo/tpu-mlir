/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace transform {
// Silenceable error propagation policy
enum class FailurePropagationMode : uint32_t {
  Propagate = 1,
  Suppress = 2,
};

::llvm::Optional<FailurePropagationMode> symbolizeFailurePropagationMode(uint32_t);
::llvm::StringRef stringifyFailurePropagationMode(FailurePropagationMode);
::llvm::Optional<FailurePropagationMode> symbolizeFailurePropagationMode(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForFailurePropagationMode() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(FailurePropagationMode enumValue) {
  return stringifyFailurePropagationMode(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<FailurePropagationMode> symbolizeEnum<FailurePropagationMode>(::llvm::StringRef str) {
  return symbolizeFailurePropagationMode(str);
}

class FailurePropagationModeAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = FailurePropagationMode;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static FailurePropagationModeAttr get(::mlir::MLIRContext *context, FailurePropagationMode val);
  FailurePropagationMode getValue() const;
};
} // namespace transform
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::transform::FailurePropagationMode> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::transform::FailurePropagationMode getEmptyKey() {
    return static_cast<::mlir::transform::FailurePropagationMode>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::transform::FailurePropagationMode getTombstoneKey() {
    return static_cast<::mlir::transform::FailurePropagationMode>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::transform::FailurePropagationMode &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::transform::FailurePropagationMode &lhs, const ::mlir::transform::FailurePropagationMode &rhs) {
    return lhs == rhs;
  }
};
}

