//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/Transforms/Passes.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"

#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>
#include <regex>
#include <sstream>

using namespace llvm;

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
    // llvm::errs() << "import calibration table:" << this->tableFile
    //              << ", is asymmetric " << this->isAsymmetric << "\n";
    auto mOp = getOperation();
    if (!module::isState(module::State::TOP_F32)) {
      mOp.dump();
      llvm_unreachable("wrong mlir state");
    }
    OpBuilder builder(mOp);
    std::map<std::string, cali_info> calibration_map;
    std::map<std::string, cali_info> calibration_map_int4;
    std::map<std::string, f64_array_t> per_chan_scales_map;
    std::ifstream infile(this->tableFile);
    if (!infile) {
      llvm_unreachable("can't open calibration table file!");
    }
    std::string line;
    std::regex cali_pattern("\\S+\\s+[-0-9.e]+\\s+[-0-9.e]+\\s+[-0-9.e]+");
    std::regex info_pattern("#.*");
    bool weight_scale_meeted = false;
    bool int4_th_meeted = false;
    while (std::getline(infile, line)) {
      if (line.back() == '\r') {
        line.pop_back();
      }

      std::istringstream iss(line);
      std::string name;
      if (weight_scale_meeted) { //third run, read weight_scale
        std::string name;
        double value;
        int num = 0;
        std::istringstream iss(line);
        iss >> name;
        iss >> num;
        auto vScales = std::make_shared<std::vector<double>>(num);
        for (int i = 0; i < num; i++) {
          iss >> value;
          vScales->data()[i] = value;
        }
        per_chan_scales_map[name] = vScales;
      } else if (int4_th_meeted) { //second run, read int4 th
        if (std::regex_match(line, cali_pattern)) {
          cali_info info = {0, 0, 0};
          if (!(iss >> name >> info.threshold >> info.min >> info.max)) {
            llvm::errs() << line;
            llvm_unreachable("\n  => not match required format\n");
          }
          calibration_map_int4[name] = info;
        } else if (std::regex_match(line, info_pattern) && std::string::npos != line.find("#weight_scale")) {
          int4_th_meeted = false;
          weight_scale_meeted = true;
        }
      } else {
        if (std::regex_match(line, cali_pattern)) { //first run, read int8 th
          cali_info info = {0, 0, 0};
          if (!(iss >> name >> info.threshold >> info.min >> info.max)) {
            llvm::errs() << line;
            llvm_unreachable("\n  => not match required format\n");
          }
          calibration_map[name] = info;
        } else if (std::regex_match(line, info_pattern) && std::string::npos != line.find("#int4_th")) {
          int4_th_meeted = true;
        }
      }
    }
    double min, max;
    for (auto func : mOp.getOps<FuncOp>()) {
      func.walk([&](Operation *op) {
        if (isa<tpu_mlir::InferenceInterface>(op) || isa<InputOp>(op)) {
          for (auto value : op->getResults()) {
            if (module::isNone(value)) {
              continue;
            }
            auto type = value.getType().cast<RankedTensorType>();
            if (type.getElementType().isIntOrIndex()) {
              continue;
            }

            auto name = module::getName(value).str();
            cali_info info = {0,0,0};
            if (calibration_map.find(name) != calibration_map.end()) {
              info = calibration_map[name];
            }
            if (calibration_map_int4.size() > 0
              && (module::isInt4Op(op) ||
              (isa<top::InputOp>(op) && module::isInt4Op(*(op->getUsers().begin()))))) {
                if (calibration_map_int4.find(name) != calibration_map_int4.end()) {
                  info = calibration_map_int4[name];
                }
            }
            if (info.max == 0 && info.min == 0 && info.threshold == 0 ) {
                continue;
            }

            getMinMax(op, info, min, max);
            auto quant_type = quant::CalibratedQuantizedType::get(
                type.getElementType(), min, max);
            auto new_type = RankedTensorType::get(type.getShape(), quant_type);
            value.setType(new_type);
          }
        } else if (isa<WeightOp>(op)) {
          auto user = op->getUsers().begin();
          std::string str = module::getName(*user).str() + "_weight";
          if (per_chan_scales_map.count(str)) {
            op->setAttr("scale", builder.getF64ArrayAttr(ArrayRef<double>{
                                     *per_chan_scales_map[str]}));
          }
        }

        if (calibration_map_int4.size() > 0 && module::isInt4Op(op)) {
          OpBuilder builder(op);
          double scale;
          int64_t zeropoint;
          auto name = module::getName(op->getResults()[0]).str();
          for (auto user : op->getUsers()) {
            if (!module::isInt4Op(user) && !isa<ReturnOp>(user)) {
              if (calibration_map.find(name) != calibration_map.end()) {
                auto &info = calibration_map[name];
                module::getScaleAndZeroPoint(info.min, info.max, scale,
                                             zeropoint);
                op->setAttr("out_int8_scale", builder.getF64FloatAttr(scale));
                op->setAttr("out_int8_zp",
                            builder.getF64FloatAttr((double)zeropoint));
              } else {
                llvm::errs() <<"not find "<< name<<"\n";
                llvm_unreachable("int4 layer's output int8 cali_info not exist\n");
              }
              break;
            }
          }

          auto preOp = op->getOperands()[0].getDefiningOp();
          if (!module::isInt4Op(preOp) && !isa<InputOp>(preOp)) {
            name = module::getName(op->getOperands()[0]).str();
            if (calibration_map_int4.find(name) != calibration_map_int4.end()) {
              auto &info = calibration_map_int4[name];
              module::getScaleAndZeroPoint(info.min, info.max, scale, zeropoint,
                                           4);
              op->setAttr("in_int4_scale", builder.getF64FloatAttr(scale));
              op->setAttr("in_int4_zp",
                          builder.getF64FloatAttr((double)zeropoint));
            } else {
              llvm::errs() <<"not find "<< name<<"\n";
              llvm_unreachable("int4 layer's input int4 cali_info not exist\n");
            }
          }
        }
      });
    }
    module::updateModuleTypes();
    module::setState(module::State::TOP_CALIBRATED);
  }
  void getMinMax(Operation *op, const cali_info &info, double &min,
                 double &max) {
    if (isa<top::AbsOp>(op)) {
      min = -info.threshold;
      max = info.threshold;
    } else if ((isa<top::SigmoidOp>(op) &&
                !dyn_cast<top::SigmoidOp>(op).getLog()) ||
               isa<top::SoftmaxOp>(op)) {
      min = 0;
      max = 1;
    } else if (isAsymmetric == false) {
      min = info.min < 0 ? (-info.threshold) : 0;
      max = info.threshold;
    } else {
      min = info.min;
      max = info.max;
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createImportCalibrationTablePass() {
  return std::make_unique<ImportCalibrationTablePass>();
}

} // namespace top
} // namespace tpu_mlir
