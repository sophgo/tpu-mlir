#include "sophgo/Dialect/Top/Transforms/Passes.h"
#include "sophgo/Support/MathUtils.h"
#include "sophgo/Support/Helper/Module.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Quant/QuantTypes.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"
#include <sstream>
#include <fstream>
#include <regex>

using namespace llvm;
using namespace mlir;
using namespace sophgo::helper;
namespace sophgo {
namespace top {

typedef struct {
  double threshold;
  double min;
  double max;
} cali_info;

class ImportCalibrationTablePass
    : public ImportCalibrationTableBase<ImportCalibrationTablePass> {
public:
  ImportCalibrationTablePass() {}
  void runOnOperation() override {
    llvm::errs() << "import calibration table:" << this->tableFile
                 << ", is asymmetric " << this->isAsymmetric << "\n";
    auto module = getOperation();
    if (!Module::isState(module, Module::State::TOP_F32)) {
      module.dump();
      llvm_unreachable("wrong mlir state");
    }
    std::map<std::string, cali_info> calibration_map;
    std::ifstream infile(this->tableFile);
    if (!infile) {
      llvm_unreachable("can't open calibration table file!");
    }
    std::string line;
    std::regex cali_pattern("\\S+\\s+[-0-9.e]+\\s+[-0-9.e]+\\s+[-0-9.e]+");
    std::regex info_pattern("#.*");
    while (std::getline(infile, line)) {
      if (line.back() == '\r') {
        line.pop_back();
      }
      std::istringstream iss(line);
      std::string name;
      if (std::regex_match(line, cali_pattern)) {
        cali_info info = {0, 0, 0};
        if (!(iss >> name >> info.threshold >> info.min >> info.max)) {
          break;
        }
        if (!isAsymmetric) {
          info.min = -info.threshold;
          info.max = info.threshold;
        }
        calibration_map[name] = info;
      } else if (std::regex_match(line, info_pattern)) {
      } else {
        // Format of threshold table error
        llvm::errs() << line;
        llvm_unreachable("\n  => not match required format\n");
      }
    }

    for (auto func : module.getOps<FuncOp>()) {
      func.walk([&](Operation *op) {
        if (isa<sophgo::InferenceInterface>(op) || isa<InputOp>(op)) {
          auto name = op->getAttrOfType<StringAttr>("name").str();
          auto value = op->getResult(0);
          auto &info = calibration_map[name];
          auto type = value.getType().cast<RankedTensorType>();
          auto quant_type = quant::CalibratedQuantizedType::get(
              type.getElementType(), info.min, info.max);
          auto new_type = RankedTensorType::get(type.getShape(), quant_type);
          value.setType(new_type);
        }
      });
    }
    Module::updateModuleTypes(module);
    Module::setState(module, Module::State::TOP_CALIBRATED);
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createImportCalibrationTablePass() {
  return std::make_unique<ImportCalibrationTablePass>();
}

} // namespace top
} // namespace sophgo
