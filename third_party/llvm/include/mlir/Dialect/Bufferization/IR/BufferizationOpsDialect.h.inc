/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace bufferization {

class BufferizationDialect : public ::mlir::Dialect {
  explicit BufferizationDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context,
      ::mlir::TypeID::get<BufferizationDialect>()) {
    
    getContext()->getOrLoadDialect<memref::MemRefDialect>();

    getContext()->getOrLoadDialect<tensor::TensorDialect>();

    initialize();
  }

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~BufferizationDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("bufferization");
  }
};
} // namespace bufferization
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::bufferization::BufferizationDialect)
