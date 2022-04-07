/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Struct Utility Definitions                                                 *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace NVVM {
MMAShapeAttr MMAShapeAttr::get(
    ::mlir::IntegerAttr m,
    ::mlir::IntegerAttr n,
    ::mlir::IntegerAttr k,
    ::mlir::MLIRContext* context) {
  ::llvm::SmallVector<::mlir::NamedAttribute, 3> fields;

  assert(m);
  auto m_id = ::mlir::StringAttr::get(context, "m");
  fields.emplace_back(m_id, m);

  assert(n);
  auto n_id = ::mlir::StringAttr::get(context, "n");
  fields.emplace_back(n_id, n);

  assert(k);
  auto k_id = ::mlir::StringAttr::get(context, "k");
  fields.emplace_back(k_id, k);

  ::mlir::Attribute dict = ::mlir::DictionaryAttr::get(context, fields);
  return dict.dyn_cast<MMAShapeAttr>();
}

bool MMAShapeAttr::classof(::mlir::Attribute attr) {
  if (!attr)
    return false;
  auto derived = attr.dyn_cast<::mlir::DictionaryAttr>();
  if (!derived)
    return false;
  int num_absent_attrs = 0;

  auto m = derived.get("m");
  if (!m || !(((m.isa<::mlir::IntegerAttr>())) && ((m.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return false;

  auto n = derived.get("n");
  if (!n || !(((n.isa<::mlir::IntegerAttr>())) && ((n.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return false;

  auto k = derived.get("k");
  if (!k || !(((k.isa<::mlir::IntegerAttr>())) && ((k.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return false;

  return derived.size() + num_absent_attrs == 3;
}

::mlir::IntegerAttr MMAShapeAttr::m() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto m = derived.get("m");
  assert(m && "attribute not found.");
  assert(m.isa<::mlir::IntegerAttr>() && "incorrect Attribute type found.");
  return m.cast<::mlir::IntegerAttr>();
}

::mlir::IntegerAttr MMAShapeAttr::n() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto n = derived.get("n");
  assert(n && "attribute not found.");
  assert(n.isa<::mlir::IntegerAttr>() && "incorrect Attribute type found.");
  return n.cast<::mlir::IntegerAttr>();
}

::mlir::IntegerAttr MMAShapeAttr::k() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto k = derived.get("k");
  assert(k && "attribute not found.");
  assert(k.isa<::mlir::IntegerAttr>() && "incorrect Attribute type found.");
  return k.cast<::mlir::IntegerAttr>();
}
} // namespace NVVM
} // namespace mlir
