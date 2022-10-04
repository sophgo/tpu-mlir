/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace utils {
// Iterator type
enum class IteratorType : uint32_t {
  parallel = 0,
  reduction = 1,
};

::llvm::Optional<IteratorType> symbolizeIteratorType(uint32_t);
::llvm::StringRef stringifyIteratorType(IteratorType);
::llvm::Optional<IteratorType> symbolizeIteratorType(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForIteratorType() {
  return 1;
}


inline ::llvm::StringRef stringifyEnum(IteratorType enumValue) {
  return stringifyIteratorType(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<IteratorType> symbolizeEnum<IteratorType>(::llvm::StringRef str) {
  return symbolizeIteratorType(str);
}
} // namespace utils
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::utils::IteratorType> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::utils::IteratorType getEmptyKey() {
    return static_cast<::mlir::utils::IteratorType>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::utils::IteratorType getTombstoneKey() {
    return static_cast<::mlir::utils::IteratorType>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::utils::IteratorType &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::utils::IteratorType &lhs, const ::mlir::utils::IteratorType &rhs) {
    return lhs == rhs;
  }
};
}

