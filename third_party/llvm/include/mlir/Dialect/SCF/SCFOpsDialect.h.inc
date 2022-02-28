/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace scf {

class SCFDialect : public ::mlir::Dialect {
  explicit SCFDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context,
      ::mlir::TypeID::get<SCFDialect>()) {
    
    getContext()->getOrLoadDialect<arith::ArithmeticDialect>();

    initialize();
  }

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~SCFDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("scf");
  }
};
} // namespace scf
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::scf::SCFDialect)
