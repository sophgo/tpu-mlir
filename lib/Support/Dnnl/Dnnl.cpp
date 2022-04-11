#include "sophgo/Support/Dnnl/Dnnl.h"
#include "sophgo/Support/Helper/Module.h"

using namespace sophgo::helper;
using namespace dnnl;

namespace sophgo {
memory::data_type getDnnlType(mlir::Value v) {
  auto type = Module::getStorageType(v);
  if (type.isF32()) {
    return memory::data_type::f32;
  }
  if (type.isSignedInteger(8) || type.isSignlessInteger(8)) {
    return memory::data_type::s8;
  }
  if (type.isUnsignedInteger(8)) {
    return memory::data_type::u8;
  }
  if (type.isInteger(16) || type.isInteger(32)) {
    return memory::data_type::s32;
  }
  llvm::errs() << "Unsupport type: ";
  type.dump();
  return memory::data_type::f32;
}
} // namespace sophgo
