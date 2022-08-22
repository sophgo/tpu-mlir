//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/Transforms/Passes.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Helper/Module.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Quant/QuantTypes.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"
#include <sstream>
#include <fstream>
#include <regex>

using namespace llvm;
using namespace mlir;
using namespace tpu_mlir::helper;
namespace tpu_mlir {
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
          info.min = info.min < 0 ? -info.threshold : 0;
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
        if (isa<tpu_mlir::InferenceInterface>(op) || isa<InputOp>(op)) {
          auto name = Module::getName(op).str();
          if (calibration_map.find(name)== calibration_map.end()) {
            llvm::errs() << "[" << name << "] not in " << this->tableFile << "!!\n";
            llvm_unreachable("Import Calibration failed!!\n");
          }
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
} // namespace tpu_mlir
