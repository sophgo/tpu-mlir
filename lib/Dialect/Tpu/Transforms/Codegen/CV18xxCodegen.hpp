//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef LIBCVIODEL_HPP_
#define LIBCVIODEL_HPP_

#include "tpu_mlir/Builder/CV18xx/cvimodel_generated.h"
#include "tpu_mlir/Builder/CV18xx/parameter_generated.h"
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Module.h"
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace cvi::model;

using FBStringVector = flatbuffers::Offset<
    flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>>;
using FBWeightVector =
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<Weight>>>;
using FBSectionVector =
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<Section>>>;
using FBTensorVector =
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<Tensor>>>;
using FBProgram = flatbuffers::Offset<Program>;
using FBRoutine = flatbuffers::Offset<Routine>;
using FBSection = flatbuffers::Offset<Section>;
using FBModel = flatbuffers::Offset<Model>;
using FBWeight = flatbuffers::Offset<Weight>;
using FBPreProcessHints = flatbuffers::Offset<PreProcessHints>;

struct op_info_t {
  Operation *op;
  uint32_t idx;
  std::string name;
  std::vector<int64_t> shape;
  size_t size;
  int64_t offset;
  DType dtype;
  bool overwrite;
  bool aligned = false;
  std::string customization_format;
  std::vector<float> scale;
  std::vector<float> bias;
};

struct debug_info_t {
  int layer_id;
  std::string name;
  bool inGroup = false;
  bool ignore = false;
  uint64_t gaddr;
  uint32_t laddr;
  std::vector<int64_t> g_shape;
  std::vector<int64_t> lg_idx_slice;
  double qscale;
  std::string qtype;
};
class CviRoutine {
public:
  CviRoutine(flatbuffers::FlatBufferBuilder &fbb, bool isTpuRoutine,
             std::string chip)
      : isTpuRoutine(isTpuRoutine), fbb_(fbb), chip(chip) {}
  virtual ~CviRoutine() {}

  std::vector<Operation *> ops;
  std::vector<Value> inputs;
  std::vector<Value> outputs;
  std::string name;
  std::string chip;
  bool isTpuRoutine;

  virtual flatbuffers::Offset<Routine> build() = 0;

protected:
  flatbuffers::FlatBufferBuilder &fbb_;
  uint32_t version_;
};

class CviTpuRoutine : public CviRoutine {
public:
  CviTpuRoutine(ModuleOp s, flatbuffers::FlatBufferBuilder &fbb,
                func::CallOp &call, int *layer_id, std::string chip);
  flatbuffers::Offset<Routine> build();

  std::vector<uint8_t> cmdbuf;
  std::vector<debug_info_t> debug_infos;
  bool isIgnore(Operation *op);
  bool isCodegen(Operation *op);

private:
  int *layer_id;
  void codeGen();
  void codegen_for_group(tpu::GroupOp gOP);
};

class CviCpuRoutine : public CviRoutine {
public:
  CviCpuRoutine(ModuleOp s, flatbuffers::FlatBufferBuilder &fbb,
                func::CallOp &call, std::string chip);
  flatbuffers::Offset<Routine> build();

private:
  Operation *op_;
  void serializeFuncArgs(std::vector<uint8_t> &args);
};

class CviModelBuilder {
public:
  CviModelBuilder(ModuleOp &module, std::string &version);
  // void storeModel(llvm::raw_ostream &output);
  void storeModel(std::string filename);

  ~CviModelBuilder() {
    for (auto &it : routines_) {
      delete it;
    }
  }
  std::vector<std::vector<debug_info_t>> tpu_debug_infos;

private:
  std::vector<uint8_t> cmdbuf;

  std::string modelName_;
  FuncOp mainFunc_;
  std::vector<CviRoutine *> routines_;
  std::vector<Operation *> ops_;
  flatbuffers::FlatBufferBuilder fbb_;
  std::vector<uint8_t> binBuffer_;
  int64_t privateGmemSize_ = 0;
  int64_t sharedGmemSize_ = 0;
  int batchNum_ = 0;
  uint32_t version_;
  uint8_t majorVersion_;
  uint8_t minorVersion_;
  uint8_t subMinorVersion_;
  int *layer_id;
  std::string chip;
  int64_t coeff_size;
  std::vector<Value> inputs;
  std::vector<Value> outputs;
  std::vector<top::WeightOp> weights;

  void addRoutine(ModuleOp s, func::CallOp &call, int *layer_id);
  FBModel build();
  FBWeightVector buildWeightMap();
  FBTensorVector buildNeuronMap();
  FBProgram buildProgram();
  FBSectionVector buildSections();
  FBSection buildSection(std::string name, cvi::model::SectionType type);
  FBSection buildSection(std::string name, cvi::model::SectionType type,
                         std::vector<uint8_t> &data);
  void parseOpInfo(Operation *op, std::string &name,
                   std::vector<int64_t> &shape, size_t &size, int64_t &offset,
                   DType &dtype, uint32_t idx);
  flatbuffers::Offset<Tensor> buildNeuron(op_info_t &op);
};

#endif // LIBCVIODEL_HPP_
