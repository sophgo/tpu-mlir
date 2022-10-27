//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include <set>
#include <memory>
#include <sstream>
#include <fstream>
#include <regex>
#include <map>
#include <elf.h>
#include <openssl/md5.h>
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/CV18xx/MlirToCvimodel.hpp"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Backend/CV18xx/CV18xx.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/Debug.h"
// #include "tpu_mlir/Support/Helper/PixeHelper.h"

#define DEBUG_TYPE "mlir-to-cvimodel"

static llvm::cl::opt<std::string>
    clModelVersion("model-version", llvm::cl::desc("cvimodel version"), llvm::cl::init("latest"));

#define VERSION(V0,V1,V2) (uint32_t)((V0) << 24 | (V1) << 16 | (V2) << 8)

using namespace tpu_mlir;
using namespace tpu_mlir::backend;
using namespace tpu_mlir::helper;
// v1.4.0
// 1) add scale/mean/pixel_format/align in Tensor
// 2) add data_format in PreProcessHints
// 3) add dmabuf in TpuRoutinea
// 4) add compress/decompress_size in Section
// 5) add mlir_version in Model
// 6) rename preprocess_hits to preprocess_hints
#define V_1_4_0 VERSION(1,4,0)

static uint32_t get_version(uint8_t &majorVersion, uint8_t &minorVersion,
                            uint8_t &subMinorVersion) {
  majorVersion = MajorVersion_value;
  minorVersion = MinorVersion_value;
  subMinorVersion = SubMinorVersion_value;
  uint32_t version = VERSION(majorVersion, minorVersion, subMinorVersion);
  return version;
}

typedef struct {
  char magic[8];
  uint32_t body_size;
  char major;
  char minor;
  char md5[16];
  char chip[16];
  char padding[2];
} CviModelHeader;


static void buildInputsOutputs(flatbuffers::FlatBufferBuilder &fbb,
                               std::vector<Value> &inputs,
                               std::vector<Value> &outputs,
                               FBStringVector &fbInputs, FBStringVector &fbOutputs) {

  std::vector<flatbuffers::Offset<flatbuffers::String>> fbStrVec;
  for (auto &v : inputs) {
    auto name = Module::getName(v).str();
    fbStrVec.push_back(fbb.CreateString(name));
  }
  fbInputs = fbb.CreateVector(fbStrVec);
  fbStrVec.clear();
  for (auto &v : outputs) {
    auto op = v.getDefiningOp();
    auto name = Module::getName(v).str();
    fbStrVec.push_back(fbb.CreateString(name));
  }
  fbOutputs = fbb.CreateVector(fbStrVec);
}

static void genMD5Hash(std::vector<uint8_t> &totalBin, uint8_t *resData) {
  MD5_CTX ctx;
  MD5_Init(&ctx);
  MD5_Update(&ctx, totalBin.data(), totalBin.size());
  MD5_Final(resData, &ctx);
}

static std::string getStrOfCurrentTime() {
  std::stringstream ssTime;
  auto clockNow = std::chrono::system_clock::now();
  auto t = std::chrono::system_clock::to_time_t(clockNow);
  ssTime << std::put_time(std::localtime(&t), "%Y-%m-%d %H:%M:%S");
  return ssTime.str();
}

CviTpuRoutine::CviTpuRoutine(flatbuffers::FlatBufferBuilder &fbb,
                             func::CallOp &call, int* layer_id,
                             std::string chip)
    : CviRoutine(fbb, true, chip) {
  this->layer_id = layer_id;
  auto func = Module::getFuncOp(Module::getModuleOp(call), call.getCallee());
  name = func.getName().str();
  func.walk([&](Operation *op) {
    if (isa<GlobalGenInterface>(op) && !Module::isOpInGroup(op)){
      ops.push_back(op);
    }
  });
  Module::getInputsOutputs(call, inputs, outputs);
  codeGen();
}

void CviTpuRoutine::codeGen() {
  auto backend_ctx = cvi_backend_create_context(chip.c_str());
  for (auto op : ops) {
    if (auto castOp = dyn_cast<tpu::GroupOp>(op)) {
      // cvi_backend_set_layer_id(backend_ctx, layer_id);
      // codegen_for_group(castOp)
      llvm_unreachable("layer group not support now");
    } else if (Module::isOpInGroup(op)) {
      // continue
    } else if (auto castOp = dyn_cast<GlobalGenInterface>(op)) {
      cvi_backend_set_layer_id(backend_ctx, *layer_id);
      castOp.codegen_global_cv18xx(backend_ctx, *layer_id);
    }
    ++(*layer_id);
    // sotre neuron
  }
  cvi_backend_submit(backend_ctx);
  cvi_backend_get_cmdbuf(backend_ctx, cmdbuf);
  cvi_backend_delete_context(backend_ctx);
}

flatbuffers::Offset<Routine> CviTpuRoutine::build() {
  FBStringVector fbInputs, fbOutputs;
  buildInputsOutputs(fbb_, inputs, outputs, fbInputs, fbOutputs);
  auto fbName = fbb_.CreateString(name);
  auto fbRoutine = CreateTpuRoutine(fbb_, fbName);
  return CreateRoutine(fbb_, RoutineType_TPU, fbInputs, fbOutputs, fbRoutine, 0);
}

CviModelBuilder::CviModelBuilder(ModuleOp &module) : fbb_(1024) {
  int layer_id = 0;
  this->chip = std::string(Module::getChip(module).lower());
  privateGmemSize_ = Module::getGmemPrivateSize(module);
  sharedGmemSize_ = Module::getNeuronSize(module);
  version_ = get_version(majorVersion_, minorVersion_, subMinorVersion_);
  modelName_ = Module::getName(module).str();
  coeff_size = Module::getCoeffSize(module); // set in assignAddr
  auto main_func = Module::getMainFuncOp(module);
  main_func.walk([&](func::CallOp call) {
    addRoutine(call, &layer_id);
  });
  Module::removeUnusedOp(module);
  for (auto func : module.getOps<FuncOp>()) {
    func.walk([&](top::WeightOp op) {
      weights.push_back(op);
    });
  }
  Module::getInputsOutputs(module, inputs, outputs);
}

void CviModelBuilder::addRoutine(func::CallOp &call, int* layer_id) {
  // todo support cpu sub_fun
  auto func = Module::getFuncOp(Module::getModuleOp(call), call.getCallee());
  bool tpu = Module::getMode(func).str() == "TPU" ? true : false;
  CviRoutine *rt = nullptr;
  if (tpu) {
    rt = new CviTpuRoutine(fbb_, call, layer_id, chip);
  } else {
    llvm_unreachable("not support now");
    // rt = new CviCpuRoutine(fbb_, mainFunc_, funcName);
  }
  routines_.push_back(rt);
}

FBSection CviModelBuilder::buildSection(std::string name,
                                        cvi::model::SectionType type) {
  auto fbName = fbb_.CreateString(name);
  uint32_t size = 0;
  uint32_t offset = 0;
  auto data_u8 = std::make_shared<std::vector<uint8_t>>(coeff_size, 0);
  for (auto weight : weights) {
    auto data = weight.read_as_byte();
    memcpy(data_u8->data() + offset, data->data(), data->size());
    offset += align_up((int64_t)data->size(), CV18xx::WEIGHT_ALIGNMENT);
    LLVM_DEBUG(llvm::errs() << "buildSection offset " << offset << "\n";);
  }
  assert(offset == coeff_size);
  uint32_t bin_offset = (uint32_t)binBuffer_.size();
  for (uint32_t i = 0; i < coeff_size; i++) {
    binBuffer_.push_back(data_u8->at(i));
    LLVM_DEBUG(llvm::errs() << "buildSection " << i << " " << (int)(data_u8->at(i)) << " end \n";);
  }
  return CreateSection(fbb_, type, fbName, coeff_size, bin_offset);
}

FBSection CviModelBuilder::buildSection(std::string name, cvi::model::SectionType type,
                                        std::vector<uint8_t>& data) {
  auto fbName = fbb_.CreateString(name);
  uint32_t size = 0;
  uint32_t offset = 0;

  size = (uint32_t)data.size();
  offset = (uint32_t)binBuffer_.size();
  // don't need compress data
  if (data.size()) {
    binBuffer_.insert(binBuffer_.end(), data.begin(), data.end());
  }
  return CreateSection(fbb_, type, fbName, size, offset);
}

FBSectionVector CviModelBuilder::buildSections() {
  std::vector<FBSection> sectionVec;
  // build weight section
  auto weightSec = buildSection("weight", SectionType_WEIGHT);
  sectionVec.push_back(weightSec);
  // build tpu cmdbuf section
  for (auto rt : routines_) {
    if (rt->isTpuRoutine) {
      auto tpuRt = (CviTpuRoutine *)rt;
      auto cmdbufSec =
          buildSection(tpuRt->name, SectionType_CMDBUF, tpuRt->cmdbuf);
      sectionVec.push_back(cmdbufSec);
    }
  }
  return fbb_.CreateVector(sectionVec);
}

FBModel CviModelBuilder::build() {
  Version modelVersion = Version(majorVersion_, minorVersion_, subMinorVersion_);
  auto fbModelName = fbb_.CreateString(modelName_);
  auto fbBuildTime = fbb_.CreateString(getStrOfCurrentTime());
  auto fbTarget = fbb_.CreateString(chip);
  auto fbMlirVersion = fbb_.CreateString(MLIR_VERSION);
  auto fbWeightMap = buildWeightMap();
  auto fbSections = buildSections();
  auto fbProgram = buildProgram();
  std::vector<FBProgram> programVec;
  programVec.push_back(fbProgram);
  auto fbProgramVec = fbb_.CreateVector(programVec);
  return CreateModel(fbb_, &modelVersion, fbModelName, fbBuildTime, 0, 0,
                     fbWeightMap, fbProgramVec, fbSections, fbTarget, fbMlirVersion);
}

void CviModelBuilder::parseOpInfo(Operation *op, std::string& name,
                                  std::vector<int64_t>& shape,
                                  size_t& size, int64_t& offset, DType& dtype) {
  auto v = op->getResult(0);
  name = Module::getName(v).str();
  auto tensorShape = Module::getShape(v);
  auto type = Module::getStorageType(v);
  auto bits = type.getIntOrFloatBitWidth();
  int dsize = -1;
  if (type.isUnsignedInteger()) {
    switch (bits) {
    case 8:
      dtype = DType::DType_UINT8;
      dsize = 1;
      break;
    case 16:
      dtype = DType::DType_UINT16;
      dsize = 2;
      break;
    case 32:
      llvm_unreachable("unsupported data type");
    default:
      break;
    }
  } else if (type.isSignedInteger() || type.isSignlessInteger()) {
    switch (bits) {
    case 8:
      dtype = DType::DType_INT8;
      dsize = 1;
      break;
    case 16:
      dtype = DType::DType_INT16;
      dsize = 2;
      break;
    case 32:
      dtype = DType::DType_INT32;
      dsize = 4;
      break;
    default:
      break;
    }
  } else if (type.isBF16()) {
    dtype = DType::DType_BF16;
    dsize = 2;
  } else if (type.isF32()) {
    dtype = DType::DType_FP32;
    dsize = 4;
  } else {
    llvm_unreachable("unsupported data type");
  }

  for (int i = 0; i < std::min((int)tensorShape.size(), 4); i++) {
    shape[i] = tensorShape[i];
  }
  if (tensorShape.size() > 4) {
    for (int i = 4; i < (int)tensorShape.size(); i++) {
      shape[3] *= tensorShape[i];
    }
  }
  size = dsize;
  for (int i = 0; i < 4; i++) {
    size *= shape[i];
  }
  // TODO if is group op get yeild addr
  offset = Module::getAddress(v);
}

flatbuffers::Offset<Tensor>
  CviModelBuilder::buildNeuron(Operation *op) {
  std::string name;
  std::vector<int64_t> shape(4, 1);
  size_t size = 0;
  int64_t offset = -1;
  DType dtype;
  if (auto castOp = dyn_cast<tpu::GroupOp>(op)) {
    // op = castOp.getRefOp(v);
    llvm_unreachable("Not support layerGroup now");
  } else if (isa<top::InputOp>(op)) {
    if (!batchNum_) {
      batchNum_ = (int)shape[0];
    }
    // TODO preprocess
  }
  if (auto castOp = dyn_cast<tpu::GroupOp>(op)) {
    // op = tpu::yield
    llvm_unreachable("Not support layerGroup now");
  }
  parseOpInfo(op, name, shape, size, offset, dtype);
  auto overwritten = false;
  // TODO set reuse flag

  // quant info
  float qscale = 0.0f; // fix me sophone set 1.0
  QuantType quant_type = QuantType_NONE;
  if (Quant::isUniformQuantized(op->getResult(0))) {
    auto qtype = Quant::getUniformQuantizedType(op->getResult(0));
    qscale = qtype.getScale();
    if (isa<top::InputOp>(op->getResult(0).getDefiningOp())) {
      qscale = 1. / qscale;
    }
  }
  auto fbShapeVec = fbb_.CreateVector(shape);
  auto fbShape = CreateShape(fbb_, fbShapeVec);
  auto fbQuant = CreateQuantInfo(fbb_, quant_type, 0, 0, 0, qscale);
  auto fbTensor = CreateTensorDirect(
    fbb_, 0, name.c_str(), offset, dtype,
    fbShape, 0, fbQuant, overwritten,
    nullptr, nullptr, nullptr, false, size);
    // TODO preprocess
    // tensor->scale.size() ? &tensor->scale : nullptr, // TODO preprocess
    // tensor->mean.size() ? &tensor->mean : nullptr,   // TODO preprocess
    // tensor->pixel_format.length() ?
    //     tensor->pixel_format.c_str() : nullptr,
    // tensor->aligned, tensor->size);
  return fbTensor;
}

FBWeightVector CviModelBuilder::buildWeightMap() {
  std::vector<FBWeight> fbWeightVec;
  for (auto &op : weights) {
    std::string name;
    std::vector<int64_t> shape(4, 1);
    int64_t offset;
    size_t size;
    DType dtype;
    parseOpInfo(op, name, shape, size, offset, dtype);
    auto fbName = fbb_.CreateString(name);
    auto fbShape = CreateShapeDirect(fbb_, &shape);
    auto fbWeight =
      CreateWeight(fbb_, fbName, offset, size, fbShape, dtype);
    fbWeightVec.push_back(fbWeight);
  }
  return fbb_.CreateVector(fbWeightVec);
}

FBTensorVector CviModelBuilder::buildNeuronMap() {
  std::vector<flatbuffers::Offset<Tensor>> tensorVec;
  for (auto v : inputs) {
    auto inputOp = v.getDefiningOp();
    tensorVec.push_back(buildNeuron(inputOp));
  }
  for (auto rt : routines_) {
    for (auto &neuronOp : rt->ops) {
      tensorVec.push_back(buildNeuron(neuronOp));
    }
  }
  return fbb_.CreateVector(tensorVec);
}

FBProgram CviModelBuilder::buildProgram() {
  auto fbNeuronMap = buildNeuronMap();

  FBStringVector fbInputs, fbOutputs;
  buildInputsOutputs(fbb_, inputs, outputs, fbInputs, fbOutputs);

  std::vector<FBRoutine> fbRoutineVec;
  for (auto rt : routines_) {
    fbRoutineVec.push_back(rt->build());
  }
  auto fbRoutines = fbb_.CreateVector(fbRoutineVec);
  return CreateProgram(fbb_, batchNum_, 0, fbInputs, fbOutputs, fbNeuronMap, fbRoutines,
                       (uint32_t)sharedGmemSize_, (uint32_t)privateGmemSize_);
}

void CviModelBuilder::storeModel(std::string filename) {
  std::string errorMessage;
  auto output = openOutputFile(filename, &errorMessage);
  if (!output) {
    llvm_unreachable(errorMessage.c_str());
  }

  FBModel fbModel = build(); // build
  fbb_.Finish(fbModel);

  std::vector<uint8_t> modelData;
  modelData.resize(fbb_.GetSize() + binBuffer_.size());
  uint8_t *dst = modelData.data();
  uint8_t *src = fbb_.GetBufferPointer();
  for (uint32_t i = 0; i < fbb_.GetSize(); i++) {
    *dst++ = *src++;
  }
  src = binBuffer_.data();
  for (uint32_t i = 0; i < binBuffer_.size(); i++) {
    *dst++ = *src++;
  }
  binBuffer_.clear();

  CviModelHeader header;
  genMD5Hash(modelData, (uint8_t *)header.md5);
  std::string magic = u8"CviModel";
  std::string padding = u8"AA";
  memcpy(header.magic, magic.c_str(), 8);
  memcpy(header.padding, magic.c_str(), 2);
  memset(header.chip, 0, sizeof(header.chip));
  strncpy(header.chip, chip.c_str(), chip.length());
  header.body_size = fbb_.GetSize();
  header.major = majorVersion_; // defined in cvimodel.fbs
  header.minor = minorVersion_; // defined in cvimodel.fbs

  output->os().write(reinterpret_cast<char *>(&header), sizeof(CviModelHeader));
  output->os().write(reinterpret_cast<char *>(modelData.data()), modelData.size());
  output->keep();
}

