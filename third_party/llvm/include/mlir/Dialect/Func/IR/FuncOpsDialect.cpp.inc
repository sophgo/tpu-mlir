/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::func::FuncDialect)
namespace mlir {
namespace func {

FuncDialect::FuncDialect(::mlir::MLIRContext *context) 
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<FuncDialect>()) {
  
    getContext()->getOrLoadDialect<cf::ControlFlowDialect>();

  initialize();
}

FuncDialect::~FuncDialect() = default;

} // namespace func
} // namespace mlir
