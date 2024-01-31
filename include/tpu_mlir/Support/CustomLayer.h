#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"

typedef union {
  int int_t;
  float float_t;
  // max size of int and float array is set as 16
  int int_arr_t[16];
  float float_arr_t[16];
} custom_param_t;

#define CUSTOM_LAYER_NAME_LEN 20
typedef struct {
  char     name[CUSTOM_LAYER_NAME_LEN + 1];
  int      param_size;
} tpu_param_t;

static void customOpProcessParam(const mlir::ArrayAttr &params, std::vector<custom_param_t> &values) {
  for (auto param : params) {
    auto dict = param.dyn_cast<mlir::DictionaryAttr>();
    for (auto element : dict) {
      mlir::Attribute value_param = element.getValue();
      custom_param_t value = {0};
      if (auto int_attr = value_param.dyn_cast<mlir::IntegerAttr>()) {
        value.int_t = int_attr.getInt();
      } else if (auto float_attr = value_param.dyn_cast<mlir::FloatAttr>()) {
        value.float_t = float_attr.getValueAsDouble();
      } else if (auto bool_attr = value_param.dyn_cast<mlir::BoolAttr>()) {
        value.int_t = bool_attr.getValue();
      } else if (auto array_attr = value_param.dyn_cast<mlir::ArrayAttr>()) {
        int num = array_attr.size();
        for (int i = 0; i < num; i++) {
          if (auto tmp_value = array_attr[i].dyn_cast<mlir::IntegerAttr>()) {
            value.int_arr_t[i] = tmp_value.getInt();
          } else if (auto tmp_value = array_attr[i].dyn_cast<mlir::FloatAttr>()) {
            value.float_arr_t[i] = tmp_value.getValueAsDouble();
          } else {
            llvm_unreachable("Only int and float vector supported now");
          }
        }
      } else {
        llvm_unreachable("Type of parameter unsupported");
      }
      values.push_back(value);
    }
  }
}
