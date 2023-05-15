/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace tosa {
// Tosa profile
enum class TosaProfileEnum : uint32_t {
  BaseInference = 0,
  MainInference = 1,
  MainTraining = 2,
  Undefined = 3,
};

::std::optional<TosaProfileEnum> symbolizeTosaProfileEnum(uint32_t);
::llvm::StringRef stringifyTosaProfileEnum(TosaProfileEnum);
::std::optional<TosaProfileEnum> symbolizeTosaProfileEnum(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForTosaProfileEnum() {
  return 3;
}


inline ::llvm::StringRef stringifyEnum(TosaProfileEnum enumValue) {
  return stringifyTosaProfileEnum(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<TosaProfileEnum> symbolizeEnum<TosaProfileEnum>(::llvm::StringRef str) {
  return symbolizeTosaProfileEnum(str);
}

class TosaProfileEnumAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = TosaProfileEnum;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static TosaProfileEnumAttr get(::mlir::MLIRContext *context, TosaProfileEnum val);
  TosaProfileEnum getValue() const;
};
} // namespace tosa
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<mlir::tosa::TosaProfileEnum, mlir::tosa::TosaProfileEnum> {
  template <typename ParserT>
  static FailureOr<mlir::tosa::TosaProfileEnum> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Tosa profile");

    // Symbolize the keyword.
    if (::std::optional<mlir::tosa::TosaProfileEnum> attr = mlir::tosa::symbolizeEnum<mlir::tosa::TosaProfileEnum>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid Tosa profile specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, mlir::tosa::TosaProfileEnum value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<mlir::tosa::TosaProfileEnum> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline mlir::tosa::TosaProfileEnum getEmptyKey() {
    return static_cast<mlir::tosa::TosaProfileEnum>(StorageInfo::getEmptyKey());
  }

  static inline mlir::tosa::TosaProfileEnum getTombstoneKey() {
    return static_cast<mlir::tosa::TosaProfileEnum>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const mlir::tosa::TosaProfileEnum &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const mlir::tosa::TosaProfileEnum &lhs, const mlir::tosa::TosaProfileEnum &rhs) {
    return lhs == rhs;
  }
};
}

