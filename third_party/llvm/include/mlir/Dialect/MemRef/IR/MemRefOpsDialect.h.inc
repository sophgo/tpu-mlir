/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace memref {

class MemRefDialect : public ::mlir::Dialect {
  explicit MemRefDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context,
      ::mlir::TypeID::get<MemRefDialect>()) {
    
    getContext()->getOrLoadDialect<arith::ArithmeticDialect>();

    initialize();
  }

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~MemRefDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("memref");
  }

  /// Materialize a single constant operation from a given attribute value with
  /// the desired resultant type.
  ::mlir::Operation *materializeConstant(::mlir::OpBuilder &builder,
                                         ::mlir::Attribute value,
                                         ::mlir::Type type,
                                         ::mlir::Location loc) override;
};
} // namespace memref
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::MemRefDialect)
