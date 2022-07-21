/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Struct Utility Declarations                                                *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace NVVM {

// Attribute for MMA operation shape.
class MMAShapeAttr : public ::mlir::DictionaryAttr {
public:
  using ::mlir::DictionaryAttr::DictionaryAttr;
  static bool classof(::mlir::Attribute attr);
  static MMAShapeAttr get(
      ::mlir::IntegerAttr m,
      ::mlir::IntegerAttr n,
      ::mlir::IntegerAttr k,
      ::mlir::MLIRContext* context);

  ::mlir::IntegerAttr m() const;
  ::mlir::IntegerAttr n() const;
  ::mlir::IntegerAttr k() const;
};

} // namespace mlir
} // namespace NVVM
