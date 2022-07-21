/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace async {

class AsyncDialect : public ::mlir::Dialect {
  explicit AsyncDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~AsyncDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("async");
  }

  /// Parse a type registered to this dialect.
  ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;

  /// Print a type registered to this dialect.
  void printType(::mlir::Type type,
                 ::mlir::DialectAsmPrinter &os) const override;

    /// The name of a unit attribute on funcs that are allowed to have a
    /// blocking async.runtime.await ops. Only useful in combination with
    /// 'eliminate-blocking-await-ops' option, which in absence of this
    /// attribute might convert a func to a coroutine.
    static constexpr StringRef kAllowedToBlockAttrName =
        "async.allowed_to_block";
  };
} // namespace async
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::async::AsyncDialect)
