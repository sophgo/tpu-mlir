//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/Transforms/Passes.h"

#include <regex>

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
    std::map<std::string, cali_info> calibration_map_fp8;
    std::ifstream infile(this->tableFile);
    if (!infile) {
      llvm_unreachable("can't open calibration table file!");
    }
    std::string line;
    std::regex cali_pattern("\\S+\\s+[-0-9.e]+\\s+[-0-9.e]+\\s+[-0-9.e]+");
    std::regex info_pattern("#.*");
    bool int8_th_meeted = true;
    bool weight_scale_meeted = false;
    bool int4_th_meeted = false;
    bool fp8_th_meeted = false;
    bool asym_op_meeted = false;
    bool int4_op_meeted = false;

    auto check_flag_line = [](std::string line, std::regex pattern) {
      // 1 for weight scale block, both fp8 and int4 int8 from mqbench
      // 2 for int4 th block
      // 3 for fp8 th block from tpu-mlir
      // 4 for fp8 th block from mqbench
      // 5 for asym_op block
      if (std::regex_match(line, pattern)) {
        if (std::string::npos != line.find("#weight_scale") ||
            std::string::npos != line.find("#weight_scale_fp8"))
          return 1;
        if (std::string::npos != line.find("#int4_th"))
          return 2;
        if (std::string::npos != line.find("#tpu-mlir-fp8"))
          return 3;
        if (std::string::npos != line.find("mqbench-fp8"))
          return 4;
        if (std::string::npos != line.find("asym_op"))
          return 5;
        if (std::string::npos != line.find("int4_op"))
          return 6;
        return 0;
      } else
        return 0;
    };

    while (std::getline(infile, line)) {
      if (line.back() == '\r') {
        line.pop_back();
      }

      std::istringstream iss(line);
      std::string name;
      std::string asym_op_name;
      std::string int4_op_name;
      auto block_type = check_flag_line(line, info_pattern);
      if (0 == block_type) {
        if (int8_th_meeted) {
          if (std::regex_match(line, cali_pattern)) { // first run, read int8 th
            cali_info info = {0, 0, 0};
            if (!(iss >> name >> info.threshold >> info.min >> info.max)) {
              llvm::errs() << line;
              llvm_unreachable("\n  => not match required format\n");
            }
            calibration_map[name] = info;
          }
        } else if (weight_scale_meeted) {
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
        } else if (int4_th_meeted) {
          if (std::regex_match(line, cali_pattern)) {
            cali_info info = {0, 0, 0};
            if (!(iss >> name >> info.threshold >> info.min >> info.max)) {
              llvm::errs() << line;
              llvm_unreachable("\n  => not match required format\n");
            }
            calibration_map_int4[name] = info;
          }
        } else if (fp8_th_meeted) {
          if (std::regex_match(line, cali_pattern)) {
            cali_info info = {0, 0, 0};
            if (!(iss >> name >> info.threshold >> info.min >> info.max)) {
              llvm::errs() << line;
              llvm_unreachable("\n  => not match required format\n");
            }
            calibration_map_fp8[name] = info;
          }
        } else if (asym_op_meeted) {
          if (!(iss >> asym_op_name)) {
            llvm::errs() << line;
            llvm_unreachable("\n  => not match required format\n");
          }
          asym_op_names.push_back(asym_op_name);
        } else if (int4_op_meeted) {
          if (!(iss >> int4_op_name)) {
            llvm::errs() << line;
            llvm_unreachable("\n  => not match required format\n");
          }
          int4_ops.push_back(int4_op_name);
        } else {
          llvm_unreachable("error th block type logic!\n");
        }
      } else {
        int8_th_meeted = block_type == 0;
        weight_scale_meeted = block_type == 1;
        int4_th_meeted = block_type == 2;
        fp8_th_meeted = ((block_type == 3) || (block_type == 4));
        asym_op_meeted = block_type == 5;
        int4_op_meeted = block_type == 6;
      }
    }
    double min, max;
    for (auto func : mOp.getOps<FuncOp>()) {
      func.walk([&](Operation *op) {
        if (isa<tpu_mlir::InferenceInterface>(op) || isa<InputOp>(op)) {
          if (isa<top::GELUOp>(op) || isa<top::SiLUOp>(op)) {
            if (std::find(asym_op_names.begin(), asym_op_names.end(),
                          module::getName(op).str()) != asym_op_names.end()) {
              op->setAttr("output_asym", builder.getBoolAttr(true));
            }
          }

          if (isa<top::ConvOp>(op) || isa<top::MatMulOp>(op)) {
            if (std::find(asym_op_names.begin(), asym_op_names.end(),
                          module::getName(op).str()) != asym_op_names.end()) {
              op->setAttr("input_asym", builder.getBoolAttr(true));
            }
          }
          for (auto value : op->getResults()) {
            if (module::isNone(value)) {
              continue;
            }
            auto type = value.getType().cast<RankedTensorType>();
            if (type.getElementType().isIntOrIndex()) {
              continue;
            }
            // if (isa<CompareConstOp>(op)) {
            //   if (dyn_cast<CompareConstOp>(op).getMode().str() != "And") {
            //     continue;
            //   }
            // }

            auto name = module::getName(value).str();
            cali_info info = {0, 0, 0};
            if (calibration_map.find(name) != calibration_map.end()) {
              info = calibration_map[name];
            }
            if (calibration_map_int4.size() > 0 &&
                (module::isInt4Op(op) ||
                 (isa<top::InputOp>(op) &&
                  module::isInt4Op(*(op->getUsers().begin())))) &&
                isOpInt4(op)) {
              if (calibration_map_int4.find(name) !=
                  calibration_map_int4.end()) {
                info = calibration_map_int4[name];
              }
            }
            if (calibration_map_fp8.find(name) != calibration_map_fp8.end()) {
              info = calibration_map_fp8[name];
              max = info.threshold;
              min = -max;
            } else {
              getMinMax(op, info, min, max);
            }
            if (module::isCV18xx()) {
              min = -max;
            }
            if (min == 0 && max == 0) {
              continue;
            }
            auto quant_type = quant::CalibratedQuantizedType::get(
                type.getElementType(), min, max);
            auto new_type = RankedTensorType::get(type.getShape(), quant_type);
            value.setType(new_type);
          }
        } else if (isa<WeightOp>(op)) {
          auto user = op->user_begin();
          std::string str = module::getName(*user).str() + "_weight";
          if (per_chan_scales_map.count(str)) {
            // use for fp8, too, but in fact it is th.
            op->setAttr("scale", builder.getF64ArrayAttr(ArrayRef<double>{
                                     *per_chan_scales_map[str]}));
          }
        }

        if (calibration_map_int4.size() > 0 && module::isInt4Op(op) &&
            isOpInt4(op)) {
          OpBuilder builder(op);
          double scale;
          int64_t zeropoint;
          auto name = module::getName(op->getResults()[0]).str();
          for (auto user : op->getUsers()) {
            if (!isOpInt4(user) && !isa<ReturnOp>(user)) {
              if (calibration_map.find(name) != calibration_map.end()) {
                auto &info = calibration_map[name];
                module::getScaleAndZeroPoint(info.min, info.max, scale,
                                             zeropoint);
                op->setAttr("out_int8_scale", builder.getF64FloatAttr(scale));
                op->setAttr("out_int8_zp",
                            builder.getF64FloatAttr((double)zeropoint));
              } else {
                llvm::errs() << "not find " << name << "\n";
                llvm_unreachable(
                    "int4 layer's output int8 cali_info not exist\n");
              }
              break;
            }
          }

          auto preOp = op->getOperands()[0].getDefiningOp();
          if (!isOpInt4(preOp) && !isa<InputOp>(preOp)) {
            name = module::getName(op->getOperands()[0]).str();
            if (calibration_map_int4.find(name) != calibration_map_int4.end()) {
              auto &info = calibration_map_int4[name];
              module::getScaleAndZeroPoint(info.min, info.max, scale, zeropoint,
                                           4);
              op->setAttr("in_int4_scale", builder.getF64FloatAttr(scale));
              op->setAttr("in_int4_zp",
                          builder.getF64FloatAttr((double)zeropoint));
            } else {
              llvm::errs() << "not find " << name << "\n";
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
    bool asymmetric = op->hasAttr("output_asym") || isAsymmetric;
    if (isa<top::AbsOp>(op)) {
      min = -info.threshold;
      max = info.threshold;
    } else if ((isa<top::SigmoidOp>(op) &&
                !dyn_cast<top::SigmoidOp>(op).getLog()) ||
               isa<top::SoftmaxOp>(op)) {
      min = 0;
      max = 1;
    } else if (isa<top::MulOp>(op)) {
      bool same = true;
      bool positive = true;
      auto in0 = op->getOperands()[0];
      for (auto in : op->getOperands()) {
        if (in0 != in) {
          same = false;
        }

        if (!isa<top::SigmoidOp, top::ReluOp>(in.getDefiningOp())) {
          positive = false;
        }
      }
      if (same || positive) {
        min = 0;
        max = info.threshold;
      } else {
        min = info.min < 0 ? (-info.threshold) : 0;
        max = info.threshold;
      }
    } else if (isa<top::SubOp>(op)) {
      min = info.min < 0 ? (-info.threshold) : -1e-5;
      max = info.threshold;
    } else if (asymmetric == false) {
      min = info.min < 0 ? (-info.threshold) : 0;
      max = info.threshold;
    } else {
      min = info.min;
      max = info.max;
    }
    if (!isa<top::SubOp>(op) && op->hasAttr("do_relu") &&
        op->getAttr("do_relu").cast<BoolAttr>().getValue()) {
      min = 0;
    }
  }

  bool isOpInt4(Operation *op) {
    auto opName = module::getName(op).str();
    return std::find(int4_ops.begin(), int4_ops.end(), opName) !=
           int4_ops.end();
  }

private:
  std::vector<std::string> asym_op_names;
  std::vector<std::string> int4_ops;
};

std::unique_ptr<OperationPass<ModuleOp>> createImportCalibrationTablePass() {
  return std::make_unique<ImportCalibrationTablePass>();
}
} // namespace top
} // namespace tpu_mlir
