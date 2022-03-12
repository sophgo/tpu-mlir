#include "sophgo/Support/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "float.h"
#include <map>
using namespace llvm;
namespace mlir {

StringRef getMlirWeightFile(ModuleOp module) {
  return module->getAttrOfType<StringAttr>("mlir.weight_file");
}

StringRef getMlirState(ModuleOp module) {
  return module->getAttrOfType<StringAttr>("mlir.state");
}

void setMlirState(mlir::ModuleOp module, StringRef state) {
  module->setAttr("mlir.state", StringAttr::get(module.getContext(), state));
}

llvm::StringRef getMlirChip(mlir::ModuleOp module) {
  return module->getAttrOfType<StringAttr>("mlir.chip");
}
void setMlirChip(mlir::ModuleOp module, StringRef chip) {
  module->setAttr("mlir.chip", StringAttr::get(module.getContext(), chip));
}

void get_scale_and_shift(float scale_f, int& scale, int& shift,int bitwidth)
{
  float min_err = FLT_MAX;
  int m_limit = (bitwidth==32) ? INT_MAX : CHAR_MAX;
  for(int n=-32;n<31;n++){ //若scale_f大于等于1，这里循环上限要设为31(而不是32)，且越大则需减少越多，暂只考虑scale_f小于等于1的情形 wxc 20220119
    int m = (int)std::round(scale_f * std::pow(2,n));
    float err = std::abs(m/std::pow(2,n) - scale_f);
    if(err < min_err && abs(m) < m_limit){
      min_err = err;
      shift = n;
    }
  }
  scale = (int)std::round(scale_f * std::pow(2,shift));
}

void get_scale_and_shift_positive(float scale_f, int& scale, int& shift,int bitwidth)
{
  float min_err = FLT_MAX;
  int m_limit = (bitwidth==32) ? INT_MAX : CHAR_MAX;
  for(int n=0;n<31;n++){
    int m = (int)std::round(scale_f * std::pow(2,n));
    float err = std::abs(m/std::pow(2,n) - scale_f);
    if(err < min_err && abs(m) < m_limit){
      min_err = err;
      shift = n;
    }
  }
  scale = (int)std::round(scale_f * std::pow(2,shift));
}

// this function search positive right shift with max_shift, max_shift set to 8 for int16 op and shift to 8bit output.
void get_scale_and_shift_positive_maxshift(float scale_f, int& scale, int& shift,int bitwidth, int max_shift)
{
  float min_err = FLT_MAX;
  int m_limit = (bitwidth==32) ? INT_MAX : CHAR_MAX;
  for(int n=0;n<max_shift;n++){
    int m = (int)std::round(scale_f * std::pow(2,n));
    float err = std::abs(m/std::pow(2,n) - scale_f);
    if(err < min_err && abs(m) < m_limit){
      min_err = err;
      shift = n;
    }
  }
  scale = (int)std::round(scale_f * std::pow(2,shift));
}

}
