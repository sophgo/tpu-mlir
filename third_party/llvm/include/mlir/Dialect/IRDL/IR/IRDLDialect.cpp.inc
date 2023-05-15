/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::irdl::IRDLDialect)
namespace mlir {
namespace irdl {

IRDLDialect::IRDLDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<IRDLDialect>()) {
  
  initialize();
}

IRDLDialect::~IRDLDialect() = default;

} // namespace irdl
} // namespace mlir
