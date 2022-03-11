//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file implements the TPU dialect OP Stats pass.
//
//===----------------------------------------------------------------------===//

#include "sophgo/Dialect/Tops/Transforms/Passes.h"
#include "sophgo/Dialect/Tops/IR/TopsOps.h"
#include "sophgo/Support/Utils.h"
#include "sophgo/Interfaces/InferenceInterface.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Quant/QuantTypes.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"
#include <sstream>
#include <fstream>
#include <regex>

using namespace llvm;

namespace mlir {
namespace tops {
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
    auto state = getMlirState(module);
    if (state != "TOPS_F32") {
      module.dump();
      llvm_unreachable("mlir state should be TOPS_F32");
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
        if (isa<mlir::InferenceInterface>(op) || isa<tops::InputOp>(op)) {
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
      // alter the function type to match the real type
      // of InputOp and ReturnOp
      std::vector<mlir::Type> arguments;
      std::vector<mlir::Type> returns;
      Block &entryBlock = func.front();
      auto returnOp = dyn_cast<func::ReturnOp>(entryBlock.back()).getOperation();
      for (uint32_t i = 0; i < entryBlock.getNumArguments(); ++i) {
        arguments.push_back(entryBlock.getArgument(i).getType());
      }
      for (uint32_t i = 0; i < returnOp->getNumOperands(); ++i) {
        returns.push_back(returnOp->getOperand(i).getType());
      }
      Builder builder(&getContext());
      auto fnType = builder.getFunctionType(llvm::ArrayRef<mlir::Type>{arguments},
                                   llvm::ArrayRef<mlir::Type>{returns});
      func.setType(fnType);
    }
    setMlirState(module, "TOPS_CALIBRATED");
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createImportCalibrationTablePass() {
  return std::make_unique<ImportCalibrationTablePass>();
}
} // namespace tops
} // namespace mlir
