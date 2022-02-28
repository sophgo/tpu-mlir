/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace cf {

class ControlFlowDialect : public ::mlir::Dialect {
  explicit ControlFlowDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context,
      ::mlir::TypeID::get<ControlFlowDialect>()) {
    
    getContext()->getOrLoadDialect<arith::ArithmeticDialect>();

    initialize();
  }

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~ControlFlowDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("cf");
  }
};
} // namespace cf
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::cf::ControlFlowDialect)
