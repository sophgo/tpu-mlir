//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "CV18xxCodegen.hpp"
#include "tpu_mlir/Backend/CV18xx/CV18xx.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/SwPipeline.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/PixelHelper.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Support/FileUtilities.h"
#include <elf.h>
#include <fstream>
#include <map>
#include <memory>
#include <openssl/md5.h>
#include <regex>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
// #include "tpu_mlir/Support/Helper/PixeHelper.h"

#define DEBUG_TYPE "mlir-to-cvimodel"
#define VERSION(V0, V1, V2) (uint32_t)((V0) << 24 | (V1) << 16 | (V2) << 8)
constexpr int64_t WEIGHT_OFFSET = (uint64_t)1 << 40;

using namespace tpu_mlir::backend;
using namespace tpu_mlir::tpu;

// v1.4.0
// 1) add scale/mean/pixel_format/align in Tensor
// 2) add data_format in PreProcessHints
// 3) add dmabuf in TpuRoutinea
// 4) add compress/decompress_size in Section
// 5) add mlir_version in Model
// 6) rename preprocess_hits to preprocess_hints
#define V_1_4_0 VERSION(1, 4, 0)

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
                               FBStringVector &fbInputs,
                               FBStringVector &fbOutputs) {

  std::vector<flatbuffers::Offset<flatbuffers::String>> fbStrVec;
  for (auto &v : inputs) {
    auto name = module::getName(v).str();
    fbStrVec.push_back(fbb.CreateString(name));
  }
  fbInputs = fbb.CreateVector(fbStrVec);
  fbStrVec.clear();
  for (auto &v : outputs) {
    auto name = module::getName(v).str();
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

CviCpuRoutine::CviCpuRoutine(flatbuffers::FlatBufferBuilder &fbb,
                             func::CallOp &call, std::string chip)
    : CviRoutine(fbb, false, chip) {
  auto func = module::getFuncOp(call.getCallee());
  func.walk([&](Operation *op) {
    if (isa<GlobalGenInterface>(op)) {
      ops.emplace_back(op);
      if (isa<GenericCpuOp>(op)) {
        op_ = op;
        name = dyn_cast<GenericCpuOp>(op).getCpuOpName().str();
      }
    }
  });
  module::getInputsOutputs(call, inputs, outputs);
}

void CviCpuRoutine::serializeFuncArgs(std::vector<uint8_t> &args) {
  flatbuffers::FlatBufferBuilder fbb(1024);
  flatbuffers::Offset<cvi::cpu_op::Attribute> attr;
  std::vector<flatbuffers::Offset<cvi::cpu_op::Attribute>> param;
  auto paramDictAttr = op_->getAttr("param").cast<DictionaryAttr>();
  for (auto &iter : paramDictAttr) {
    auto key = iter.getName().str();
    auto flatKey = fbb.CreateString(key);
    if (iter.getValue().isa<StringAttr>()) {
      auto value = iter.getValue().cast<StringAttr>().getValue();
      std::string strValue = std::string(value.data(), value.size());
      auto flatValue = fbb.CreateString(strValue);
      auto strAttr = cvi::cpu_op::CreateStrAttr(fbb, flatKey, flatValue);
      attr = cvi::cpu_op::CreateAttribute(fbb, 0, 0, 0, strAttr, 0, 0);
    } else if (iter.getValue().isa<BoolAttr>()) {
      auto value = iter.getValue().cast<BoolAttr>().getValue();
      auto boolAttr = cvi::cpu_op::CreateBoolAttr(fbb, flatKey, value);
      attr = cvi::cpu_op::CreateAttribute(fbb, 0, boolAttr, 0, 0, 0, 0);
    } else if (iter.getValue().isa<IntegerAttr>()) {
      auto value = iter.getValue().cast<IntegerAttr>().getInt();
      auto intAttr = cvi::cpu_op::CreateIntAttr(fbb, flatKey, value);
      attr = cvi::cpu_op::CreateAttribute(fbb, 0, 0, intAttr, 0, 0, 0);
    } else if (iter.getValue().isa<FloatAttr>()) {
      auto value = iter.getValue().cast<FloatAttr>().getValueAsDouble();
      auto floatAttr = cvi::cpu_op::CreateFloatAttr(fbb, flatKey, value);
      attr = cvi::cpu_op::CreateAttribute(fbb, floatAttr, 0, 0, 0, 0, 0);
    } else if (iter.getValue().isa<DenseFPElementsAttr>()) {
      std::vector<float> fpArray;
      auto value = iter.getValue().cast<DenseFPElementsAttr>();
      for (APFloat realVal : value) {
        fpArray.push_back(realVal.convertToFloat());
      }
      auto flatValue = fbb.CreateVector(fpArray);
      auto fpArrayAttr =
          cvi::cpu_op::CreateFloatArrayAttr(fbb, flatKey, flatValue);
      attr = cvi::cpu_op::CreateAttribute(fbb, 0, 0, 0, 0, fpArrayAttr, 0);
    } else if (iter.getValue().isa<DenseIntElementsAttr>()) {
      std::vector<int> intArray;
      auto value = iter.getValue().cast<DenseIntElementsAttr>();
      for (APInt intVal : value) {
        intArray.push_back(intVal.getZExtValue());
      }
      auto flatValue = fbb.CreateVector(intArray);
      auto intArrayAttr =
          cvi::cpu_op::CreateIntArrayAttr(fbb, flatKey, flatValue);
      attr = cvi::cpu_op::CreateAttribute(fbb, 0, 0, 0, 0, 0, intArrayAttr);
    } else if (iter.getValue().isa<ArrayAttr>()) {
      auto value = iter.getValue().cast<ArrayAttr>();
      if ((*value.begin()).dyn_cast_or_null<IntegerAttr>()) {
        std::vector<int> intArray;

        for (auto &intVal : value) {
          intArray.push_back(intVal.cast<IntegerAttr>().getInt());
        }
        auto flatValue = fbb.CreateVector(intArray);
        auto intArrayAttr =
            cvi::cpu_op::CreateIntArrayAttr(fbb, flatKey, flatValue);
        attr = cvi::cpu_op::CreateAttribute(fbb, 0, 0, 0, 0, 0, intArrayAttr);
      } else {
        llvm_unreachable("unsupported type, only support i32 array parsing");
      }
    } else {
      llvm_unreachable("unsupported type");
    }
    param.push_back(attr);
  }

  auto fbParam = cvi::cpu_op::CreateParameterDirect(fbb, &param);
  fbb.Finish(fbParam);

  uint8_t *ptr = fbb.GetBufferPointer();
  for (uint32_t i = 0; i < fbb.GetSize(); i++) {
    args.push_back(*ptr++);
  }
}

flatbuffers::Offset<Routine> CviCpuRoutine::build() {
  FBStringVector fbInputs, fbOutputs;
  // For some cpu functions, weightOp maybe the operand and the weightOp's name
  // should be added to the inputs.
  inputs.clear();
  for (uint32_t i = 0; i < op_->getNumOperands(); i++) {
    auto v = module::getOperand(op_, i);
    inputs.push_back(v);
  }
  buildInputsOutputs(fbb_, inputs, outputs, fbInputs, fbOutputs);
  std::vector<uint8_t> args;
  serializeFuncArgs(args);
  auto fbName = fbb_.CreateString(name);
  auto fbRoutine = CreateCpuRoutineDirect(fbb_, name.c_str(), &args);
  return CreateRoutine(fbb_, RoutineType_CPU, fbInputs, fbOutputs, 0,
                       fbRoutine);
}

CviTpuRoutine::CviTpuRoutine(flatbuffers::FlatBufferBuilder &fbb,
                             func::CallOp &call, int *layer_id,
                             std::string chip)
    : CviRoutine(fbb, true, chip) {
  this->layer_id = layer_id;
  auto func = module::getFuncOp(call.getCallee());
  name = func.getName().str();
  func.walk([&](Operation *op) {
    if (isa<GlobalGenInterface, GroupOp>(op) && !module::isOpInGroup(op)) {
      ops.push_back(op);
    }
  });
  module::getInputsOutputs(call, inputs, outputs);
  codeGen();
}

void CviTpuRoutine::codegen_for_group(GroupOp gOp) {
  auto nsecs = gOp.getNsecs();
  auto hsecs = gOp.getHsecs();
  auto swpipl_stage_num = gOp.getSwpiplStageNum();
  auto &body = gOp.getBody().front();
  auto flow = module::getI64Array(gOp.getFlow());
  // 1. restore timestep_table from flow
  std::vector<std::vector<int64_t>> timestep_table;
  std::vector<int64_t> ts_row;
  int64_t max_id = 0;
  for (size_t i = 1; i < flow->size(); ++i) {
    if (flow->at(i) < 0) {
      timestep_table.push_back(ts_row);
      ts_row.clear();
      continue;
    }
    ts_row.push_back(flow->at(i));
    max_id = std::max(max_id, flow->at(i));
  }
  timestep_table.push_back(ts_row);
  int timestep_num = timestep_table.size();
  // 2. create a vector to map id to op
  std::vector<Operation *> group_ops;
  for (int64_t id = 0; id < max_id;) {
    body.walk([&](Operation *op) {
      if (auto lgOp = dyn_cast<LocalGenInterface>(op)) {
        auto ginfo = lgOp.getGroupInfo((int64_t)0, (int64_t)0, (int64_t)0, (int64_t)0);
        if (ginfo.id == id) {
          group_ops.push_back(op);
          id++;
        }
      }
    });
  }
  // 3. codegen for group
  int64_t stage_idx = 0;
  int64_t draining_idx = 0;
  bool draining_period = false;
  SoftwarePipeline timestep_swpipl;
  for (uint64_t nstep = 0, hstep = 0; nstep < nsecs || draining_period;) {
    /* add for software pipeline */
    timestep_swpipl.write_swloop_buffer(nstep, hstep, 0, 0, swpipl_stage_num);
    for (uint32_t ts = 0; ts < timestep_num; ++ts) {
      CV18xx::parallel_enable();
      auto cur_op_ids = timestep_table[ts];
      for (auto id : cur_op_ids) {
        auto lgOp = cast<LocalGenInterface>(group_ops[id]);
        auto ginfo = lgOp.getGroupInfo(nstep, hstep, 0, 0);
        if ((!draining_period && ginfo.stage > stage_idx) ||
            (draining_period &&
             (ginfo.stage < draining_idx || ginfo.stage > stage_idx))) {
          continue;
        }
        const tensor_step_t *tensor_step =
            timestep_swpipl.read_swloop_buffer(ginfo.stage);
        ginfo = lgOp.getGroupInfo(tensor_step->nstep, tensor_step->hstep, tensor_step->dstep, tensor_step->wstep);

        // add prefix to each cmd in profile.txt
        std::string prefix = module::getName(group_ops[id]).str();
        if (ginfo.overstepped == false) {
          CV18xx::set_layer_id(*layer_id);
          lgOp.codegen_local_cv18xx(tensor_step->nstep, tensor_step->hstep,
                                    *layer_id);
          ++(*layer_id);
        }
      } // ops, include Load/Store op
      CV18xx::parallel_disable();
    } // timestep

    if (!draining_period) {
      hstep++;
      if (hstep >= hsecs) {
        hstep = 0;
        nstep++;
        if (nstep >= nsecs) {
          draining_period = true;
        }
      }
    }
    if (draining_period) {
      draining_idx++;
      if (draining_idx >= swpipl_stage_num) {
        draining_period = false;
      }
    }
    stage_idx++;
  }
}

void CviTpuRoutine::codeGen() {
  for (auto op : ops) {
    if (auto castOp = dyn_cast<GroupOp>(op)) {
      codegen_for_group(castOp);
    } else if (module::isOpInGroup(op)) {
      continue;
    } else if (auto castOp = dyn_cast<GlobalGenInterface>(op)) {
      CV18xx::set_layer_id(*layer_id);
      castOp.codegen_global_cv18xx(*layer_id);
      ++(*layer_id);
    }
    // sotre neuron
  }
  CV18xx::submit();
  CV18xx::read_cmdbuf(cmdbuf);
}

flatbuffers::Offset<Routine> CviTpuRoutine::build() {
  FBStringVector fbInputs, fbOutputs;
  buildInputsOutputs(fbb_, inputs, outputs, fbInputs, fbOutputs);
  auto fbName = fbb_.CreateString(name);
  auto fbRoutine = CreateTpuRoutine(fbb_, fbName);
  return CreateRoutine(fbb_, RoutineType_TPU, fbInputs, fbOutputs, fbRoutine,
                       0);
}

CviModelBuilder::CviModelBuilder(ModuleOp &module) : fbb_(1024) {
  int layer_id = 0;
  auto chip_ = module::getChip();
  chip = module::stringifyChip(chip_);
  privateGmemSize_ = module::getGmemPrivateSize();
  sharedGmemSize_ = module::getNeuronSize();
  version_ = get_version(majorVersion_, minorVersion_, subMinorVersion_);
  modelName_ = module::getModuleName().str();
  coeff_size = module::getCoeffSize(); // set in assignAddr
  auto main_func = module::getMainFuncOp();
  main_func.walk([&](func::CallOp call) { addRoutine(call, &layer_id); });
  for (auto func : module.getOps<FuncOp>()) {
    func.walk([&](top::WeightOp op) { weights.push_back(op); });
  }
  module::getInputsOutputs(inputs, outputs);
}

void CviModelBuilder::addRoutine(func::CallOp &call, int *layer_id) {
  // todo support cpu sub_fun
  auto func = module::getFuncOp(call.getCallee());
  auto run_mode = getRunMode(func);
  CviRoutine *rt = nullptr;
  switch (run_mode) {
  case RunMode::TPU_STATIC:
    rt = new CviTpuRoutine(fbb_, call, layer_id, chip);
    break;
  case RunMode::CPU:
    rt = new CviCpuRoutine(fbb_, call, chip);
    break;
  default:
    llvm_unreachable("Not Implemented");
    break;
  }
  routines_.push_back(rt);
}

FBSection CviModelBuilder::buildSection(std::string name,
                                        cvi::model::SectionType type) {
  auto fbName = fbb_.CreateString(name);
  auto data_u8 = std::make_shared<std::vector<uint8_t>>(coeff_size, 0);
  std::unordered_set<int64_t> weight_set;
  for (auto weight : weights) {
    int64_t offset = module::getAddress(weight.getOutput());
    if (weight_set.find(offset) != weight_set.end()) {
      continue;
    } else {
      weight_set.emplace(offset);
    }
    offset -= WEIGHT_OFFSET;
    auto data = weight.read_as_byte();
    memcpy(data_u8->data() + offset, data->data(), data->size());
  }
  uint32_t bin_offset = (uint32_t)binBuffer_.size();
  std::copy(data_u8->begin(), data_u8->end(), std::back_inserter(binBuffer_));
  return CreateSection(fbb_, type, fbName, coeff_size, bin_offset);
}

FBSection CviModelBuilder::buildSection(std::string name,
                                        cvi::model::SectionType type,
                                        std::vector<uint8_t> &data) {
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
      if (tpuRt->cmdbuf.size()) {
        auto cmdbufSec =
            buildSection(tpuRt->name, SectionType_CMDBUF, tpuRt->cmdbuf);
        sectionVec.push_back(cmdbufSec);
      }
    }
  }
  return fbb_.CreateVector(sectionVec);
}

FBModel CviModelBuilder::build() {
  Version modelVersion =
      Version(majorVersion_, minorVersion_, subMinorVersion_);
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
                     fbWeightMap, fbProgramVec, fbSections, fbTarget,
                     fbMlirVersion);
}

void CviModelBuilder::parseOpInfo(Operation *op, std::string &name,
                                  std::vector<int64_t> &shape, size_t &size,
                                  int64_t &offset, DType &dtype, uint32_t idx) {
  auto v = op->getResult(idx);
  name = module::getName(v).str();
  auto tensorShape = module::getShape(v);
  if (auto castOp = llvm::dyn_cast<top::InputOp>(op)) {
    // input op should use the argshape
    tensorShape = module::getShape(op->getOperand(0));
  }
  auto type = module::getStorageType(v);
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
  offset = module::getAddress(v);
}

static int getDsize(DType &dtype) {
  int dsize = -1;
  if (dtype == DType::DType_INT8 || dtype == DType::DType_UINT8 ||
      dtype == DType::DType_MAX) {
    dsize = 1;
  } else if (dtype == DType::DType_BF16 || dtype == DType::DType_INT16 ||
             dtype == DType::DType_UINT16) {
    dsize = 2;
  } else if (dtype == DType::DType_INT32 || dtype == DType::DType_FP32 ||
             dtype == DType::DType_MIN) {
    dsize = 4;
  } else {
    llvm_unreachable("unsupported data type");
  }
  return dsize;
}

flatbuffers::Offset<Tensor> CviModelBuilder::buildNeuron(op_info_t &op_info) {

  // quant info
  float qscale = 0.0f; // fix me sophone set 1.0
  QuantType quant_type = QuantType_NONE;
  if (module::isUniformQuantized(op_info.op->getResult(op_info.idx))) {
    auto qtype =
        module::getUniformQuantizedType(op_info.op->getResult(op_info.idx));
    qscale = qtype.getScale();
    if (isa<top::InputOp>(op_info.op->getResult(0).getDefiningOp())) {
      qscale = 1. / qscale;
    }
  }
  auto fbShapeVec = fbb_.CreateVector(op_info.shape);
  auto fbShape = CreateShape(fbb_, fbShapeVec);
  auto fbQuant = CreateQuantInfo(fbb_, quant_type, 0, 0, 0, qscale);
  auto fbTensor =
      CreateTensorDirect(fbb_, 0, op_info.name.c_str(), op_info.offset,
                         op_info.dtype, fbShape, 0, fbQuant, op_info.overwrite,
                         op_info.scale.size() ? &op_info.scale : nullptr,
                         op_info.bias.size() ? &op_info.bias : nullptr,
                         op_info.customization_format.length()
                             ? op_info.customization_format.c_str()
                             : nullptr,
                         op_info.aligned, op_info.size);
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
    parseOpInfo(op, name, shape, size, offset, dtype, 0);
    auto fbName = fbb_.CreateString(name);
    auto fbShape = CreateShapeDirect(fbb_, &shape);
    auto fbWeight = CreateWeight(fbb_, fbName, offset, size, fbShape, dtype);
    fbWeightVec.push_back(fbWeight);
  }
  return fbb_.CreateVector(fbWeightVec);
}

void markGmemReusedOp(std::vector<op_info_t> &ops,
                      std::set<op_info_t *> &gmemReusedSet) {
  std::vector<op_info_t *> tmp;
  for (int i = ops.size() - 1; i >= 0; i--) {
    auto addr_i = ops[i].offset;
    auto sz_i = ops[i].size;
    for (int j = 0; j < (int)tmp.size(); j++) {
      auto addr_j = tmp[j]->offset;
      auto sz_j = tmp[j]->size;
      auto start = std::min(addr_i, addr_j);
      auto end = std::max(addr_i + sz_i, addr_j + sz_j);
      // memory overlap
      if (end - start < sz_i + sz_j) {
        gmemReusedSet.insert(&ops[i]);
      }
    }
    tmp.emplace_back(&ops[i]);
  }
}

FBTensorVector CviModelBuilder::buildNeuronMap() {
  std::vector<flatbuffers::Offset<Tensor>> tensorVec;
  std::vector<op_info_t> ops;
  for (auto v : inputs) {
    auto inputOp = v.getDefiningOp();
    op_info_t op_info;
    op_info.op = inputOp;
    op_info.overwrite = false;
    op_info.shape.resize(4, 1);
    op_info.idx = 0;
    op_info.aligned = false;
    parseOpInfo(inputOp, op_info.name, op_info.shape, op_info.size,
                op_info.offset, op_info.dtype, op_info.idx);
    if (auto castOp = llvm::dyn_cast<top::InputOp>(inputOp)) {
      // fuse preprocess and aligned_input
      if (castOp.getCustomizationFormat()) {
        auto scale = module::getF64Array(castOp.getScaleAttr());
        auto mean = module::getF64Array(castOp.getMeanAttr());
        op_info.customization_format =
            castOp.getCustomizationFormatAttr().str();
        for (int i = 0; i < scale->size(); i++) {
          op_info.scale.emplace_back(scale->at(i));
          op_info.bias.emplace_back(scale->at(i) * mean->at(i));
        }
        if (castOp.getAligned()) {
          op_info.aligned = castOp.getAligned().value();
        }
        if (op_info.aligned) {
          int64_t y_align, w_align, channel_align;
          setPixelAlign(op_info.customization_format, y_align, w_align,
                        channel_align);
          int dsize = getDsize(op_info.dtype);
          op_info.size =
              dsize * aligned_image_size(op_info.shape[0], op_info.shape[1],
                                         op_info.shape[2], op_info.shape[3],
                                         op_info.customization_format, y_align,
                                         w_align, channel_align);
          llvm::errs() << chip << " input tensor[" << op_info.shape[0] << ", "
                       << op_info.shape[1] << "," << op_info.shape[2] << ","
                       << op_info.shape[3]
                       << "]  pixel_format: " << op_info.customization_format
                       << "  y aligned:" << y_align << "  w aligned:" << w_align
                       << "  c aligned:" << channel_align
                       << "  tensor size:" << op_info.size
                       << " tensor dtype:" << op_info.dtype
                       << "  dsize:" << dsize << "\n";
        }
      }
    }
    ops.emplace_back(op_info);
  }
  for (auto rt : routines_) {
    for (auto &neuronOp : rt->ops) {
      for (uint32_t i = 0; i < neuronOp->getNumResults(); ++i) {
        if (!module::isNone(neuronOp->getResults()[i])) {
          op_info_t op_info;
          op_info.op = neuronOp;
          op_info.overwrite = false;
          op_info.shape.resize(4, 1);
          op_info.idx = i;
          parseOpInfo(neuronOp, op_info.name, op_info.shape, op_info.size,
                      op_info.offset, op_info.dtype, op_info.idx);
          ops.emplace_back(op_info);
        }
      }
    }
  }
  std::set<op_info_t *> op_reused;
  markGmemReusedOp(ops, op_reused);
  for (auto &op_info : ops) {
    if (op_reused.find(&op_info) != op_reused.end()) {
      op_info.overwrite = true;
    }
    tensorVec.emplace_back(buildNeuron(op_info));
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
  return CreateProgram(fbb_, batchNum_, 0, fbInputs, fbOutputs, fbNeuronMap,
                       fbRoutines, (uint32_t)sharedGmemSize_,
                       (uint32_t)privateGmemSize_);
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
  output->os().write(reinterpret_cast<char *>(modelData.data()),
                     modelData.size());
  output->keep();
}
