/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES


namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir
namespace mlir {
namespace async {
class CoroHandleType;
class CoroIdType;
class CoroStateType;
class GroupType;
class TokenType;
class ValueType;
class CoroHandleType : public ::mlir::Type::TypeBase<CoroHandleType, ::mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
public:
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"coro.handle"};
  }

};
class CoroIdType : public ::mlir::Type::TypeBase<CoroIdType, ::mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
public:
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"coro.id"};
  }

};
class CoroStateType : public ::mlir::Type::TypeBase<CoroStateType, ::mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
public:
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"coro.state"};
  }

};
class GroupType : public ::mlir::Type::TypeBase<GroupType, ::mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
public:
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"group"};
  }

};
class TokenType : public ::mlir::Type::TypeBase<TokenType, ::mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
public:
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"token"};
  }

};
namespace detail {
struct ValueTypeStorage;
} // namespace detail
class ValueType : public ::mlir::Type::TypeBase<ValueType, ::mlir::Type, detail::ValueTypeStorage> {
public:
  using Base::Base;
public:
  static ValueType get(Type valueType);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"value"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  Type getValueType() const;
};
} // namespace async
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::async::CoroHandleType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::async::CoroIdType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::async::CoroStateType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::async::GroupType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::async::TokenType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::async::ValueType)

#endif  // GET_TYPEDEF_CLASSES

