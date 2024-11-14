#include "assert.h"
#include "md5.hpp"
#include "tpu_mlir/Builder/CV18xx/cvimodel_generated.h"
#include "tpu_mlir/Builder/CV18xx/parameter_generated.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <math.h>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

namespace cvtool {

constexpr int64_t WEIGHT_OFFSET = (uint64_t)1 << 40;

class FileStream;
using FBWeightVector = flatbuffers::Offset<
    flatbuffers::Vector<flatbuffers::Offset<cvi::model::Weight>>>;
using FBTensorVector = flatbuffers::Offset<
    flatbuffers::Vector<flatbuffers::Offset<cvi::model::Tensor>>>;
using FBRoutineVector = flatbuffers::Offset<
    flatbuffers::Vector<flatbuffers::Offset<cvi::model::Routine>>>;
using FBProgramVector = flatbuffers::Offset<
    flatbuffers::Vector<flatbuffers::Offset<cvi::model::Program>>>;
using FBSectionVector = flatbuffers::Offset<
    flatbuffers::Vector<flatbuffers::Offset<cvi::model::Section>>>;
using SectionInfoVector =
    std::vector<std::tuple<FileStream *, uint32_t, uint32_t, uint32_t>>;

// from cviruntime model.hpp
struct MODEL_HEADER {
  char magic[8];
  uint32_t body_size;
  char major;
  char minor;
  char md5[16];
  char chip[16];
  char padding[2];
};

// from cviruntime stream.hpp
class FileStream {
public:
  FileStream(const std::string &file_name) {
    _fstream = new std::ifstream(file_name, std::ifstream::binary);
    if (!_fstream->good()) {
      printf("Error, Failed to open %s\n", file_name.c_str());
      return;
    }
    _fstream->seekg(0, _fstream->end);
    _length = _fstream->tellg();
    _fstream->seekg(0, _fstream->beg);
  }
  ~FileStream() {
    if (_fstream) {
      delete _fstream;
    }
  }
  size_t length() { return _length; }
  size_t read(uint8_t *buf, size_t offset, size_t size) {
    assert(offset + size <= _length && "model is incomplete or incorrect!");
    _fstream->seekg(offset);
    _fstream->read((char *)buf, size);
    return size;
  }

private:
  std::ifstream *_fstream;
  size_t _length;
};

class Model {
public:
  Model(const std::string &model_file);
  ~Model() {
    if (fb_model_buffer)
      delete[] fb_model_buffer;
  }

  void dump();
  void extract();
  void merge(std::vector<std::shared_ptr<Model>> &models, std::string &dst);
  void show_chip();

  MODEL_HEADER header;
  FileStream stream;
  size_t binary_offset;
  const cvi::model::Model *fb_model;

private:
  uint8_t *fb_model_buffer;
  flatbuffers::FlatBufferBuilder fbb;
  float _qscale = 0;
  std::string input_quanted_tensor;

  void storeSectionToFile(const cvi::model::Section *section, std::string dst);
  std::string calcSectionMD5(const cvi::model::Section *section, int size);
  std::string calcMD5(std::vector<uint8_t> &data, int64_t size);
  int64_t extractWeightData(const cvi::model::Section *section,
                            int64_t weight_offset, int64_t size,
                            std::vector<uint8_t> &weight_data);
  const char *sectionTypeToStr(cvi::model::SectionType type);
  const char *dtypeToStr(cvi::model::DType type);
  size_t dtypeSize(cvi::model::DType type);
  bool getQscaleFromDequantCpuOp(const cvi::cpu_op::Parameter *param);

  void dumpBaseInfo();
  void dumpSections();
  void dumpWeightMap();
  void dumpPrograms();
  void dumpTensors(const cvi::model::Program *p, bool oldVersion);
  void dumpRoutines(const cvi::model::Program *p);

  FBWeightVector
  cloneWeightMap(std::vector<std::shared_ptr<Model>> &models,
                 std::map<int64_t, std::vector<uint8_t>> &weight_data_map);
  FBTensorVector cloneTensorMap(const cvi::model::Program *program);
  FBRoutineVector
  cloneRoutines(const cvi::model::Program *program, bool rename, int index,
                std::map<std::string, std::string> *routine_name_map);
  FBProgramVector clonePrograms(
      std::vector<std::shared_ptr<Model>> &models,
      std::vector<std::map<std::string, std::string>> &routine_name_maps);
  FBSectionVector cloneSections(
      std::vector<std::shared_ptr<Model>> &models,
      std::vector<uint8_t> &sections_buf,
      std::vector<std::map<std::string, std::string>> &routine_name_maps,
      std::map<int64_t, std::vector<uint8_t>> &weight_data_map);
};

Model::Model(const std::string &model_file) : stream(model_file), fbb(0) {
  if (stream.length() <= sizeof(header)) {
    printf("Error, invalid cvimodel file\n");
  }
  stream.read((uint8_t *)&header, 0, sizeof(header));
  size_t header_size;
  /* before version 1.1, heder size is 32 bytes */
  if (header.major == 1 && header.minor == 0)
    header_size = 0x20;
  else
    header_size = sizeof(MODEL_HEADER);
  binary_offset = header_size + header.body_size;
  fb_model_buffer = new uint8_t[header.body_size];
  if (!fb_model_buffer) {
    printf("Failed to allocate memory\n");
  }
  stream.read(fb_model_buffer, header_size, header.body_size);
  fb_model = cvi::model::GetModel(fb_model_buffer);
}

void Model::dump() {
  dumpBaseInfo();
  dumpSections();
  dumpWeightMap();
  dumpPrograms();
}

void Model::extract() {
  auto name = fb_model->name()->str();
  for (auto p : *fb_model->programs()) {
    auto neuron_size = p->neuron_size();
    auto private_gmem_size = p->private_gmem();
    auto shared_gmem_size = p->shared_gmem();
    if (neuron_size) {
      std::cout << "neuron_size: " << neuron_size << "\n";
    } else {
      std::cout << "private_mem_size: " << private_gmem_size << "\n";
      std::cout << "shared_gmem_size: " << shared_gmem_size << "\n";
    }
    // dump cmdbuf
    for (auto r : *p->routines()) {
      if (r->type() != cvi::model::RoutineType_TPU)
        continue;

      auto getTensorByName = [&](std::string name) {
        for (const auto &t : *p->tensor_map()) {
          if (t->name()->str() == name) {
            return t;
          }
        }
        assert(0);
      };

      std::string buf_name;
      std::string buf_type_name;
      if (r->tpu_routine()->cmdbuf_section()) {
        buf_name = r->tpu_routine()->cmdbuf_section()->str();
        buf_type_name = "cmdbuf";
      } else if (r->tpu_routine()->dmabuf_section()) {
        buf_name = r->tpu_routine()->dmabuf_section()->str();
        buf_type_name = "dmabuf";
      } else {
        assert(0 && "model has not cmdbuf and dmabuf");
      }

      printf("routine #%s\n", buf_name.c_str());
      printf("  %-6s %-4s %-4s %-4s %-4s %-5s %-7s %s\n", " ", "n", "c", "h",
             "w", "dtype", "offset", "name");
      for (auto name : *r->in_tensors()) {
        auto tensor = getTensorByName(name->str());
        auto &shape = *tensor->shape()->dim();
        printf("  %-6s %-4d %-4d %-4d %-4d %-5s %-7d %s\n", "[IN ]",
               (int)shape[0], (int)shape[1], (int)shape[2], (int)shape[3],
               dtypeToStr(tensor->dtype()), (int)tensor->offset(),
               name->c_str());
      }
      for (auto name : *r->out_tensors()) {
        auto tensor = getTensorByName(name->str());
        auto &shape = *tensor->shape()->dim();
        printf("  %-6s %-4d %-4d %-4d %-4d %-5s %-7d %s\n", "[OUT]",
               (int)shape[0], (int)shape[1], (int)shape[2], (int)shape[3],
               dtypeToStr(tensor->dtype()), (int)tensor->offset(),
               name->c_str());
      }
      for (auto s : *fb_model->sections()) {
        if (s->name()->str() == buf_name) {
          std::string dst = name + "_program_" +
                            std::to_string(p->batch_num()) + "_" +
                            buf_type_name + "_" + s->name()->str() + ".bin";
          storeSectionToFile(s, dst);
          break;
        }
      }
    }
  }
  // dump weight
  for (auto s : *fb_model->sections()) {
    if (s->type() != cvi::model::SectionType_WEIGHT)
      continue;
    storeSectionToFile(s, name + "_weight.bin");
  }
}

void Model::storeSectionToFile(const cvi::model::Section *section,
                               std::string dst) {
  auto offset = section->offset();
  auto size = section->size();
  std::ofstream of(dst, std::ofstream::out | std::ofstream::binary |
                            std::ofstream::trunc);
  if (section->compress()) {
    assert(0 && "not supported!");
  } else {
    uint8_t *buf = new uint8_t[1024];
    do {
      auto len = size > 1024 ? 1024 : size;
      stream.read(buf, binary_offset + offset, len);
      of.write((const char *)buf, len);
      offset += len;
      size -= len;
    } while (size);
    of.close();
    delete[] buf;
  }
  printf("store section to %s\n", dst.c_str());
}

std::string Model::calcSectionMD5(const cvi::model::Section *section,
                                  int size) {
  auto offset = section->offset();
  MD5 md5;
  if (section->compress()) {
    assert(0 && "not supported!");
  } else {
    uint8_t *buf = new uint8_t[1024];
    do {
      auto len = size > 1024 ? 1024 : size;
      stream.read(buf, binary_offset + offset, len);
      md5.update(buf, len);
      offset += len;
      size -= len;
    } while (size);
    delete[] buf;
  }
  return md5.finalize().hexdigest();
}

std::string Model::calcMD5(std::vector<uint8_t> &data, int64_t size) {
  MD5 md5;
  md5.update(data.data(), size);
  return md5.finalize().hexdigest();
}

int64_t Model::extractWeightData(const cvi::model::Section *section,
                                 int64_t weight_offset, int64_t size,
                                 std::vector<uint8_t> &weight_data) {
  auto offset = section->offset();
  if (section->compress()) {
    assert(0 && "not supported!");
  }
  weight_data.resize(size);
  return stream.read(weight_data.data(), binary_offset + offset + weight_offset,
                     size);
}

void Model::show_chip() {
  if (fb_model->target()) {
    printf("For %s chip ONLY\n", fb_model->target()->c_str());
  } else {
    printf("No chip Info\n");
  }
}

void Model::dumpBaseInfo() {
  if (fb_model->mlir_version()) {
    printf("Mlir Version: %s\n", fb_model->mlir_version()->c_str());
  }
  auto version = fb_model->version();
  printf("Cvimodel Version: %d.%d.%d\n", (int)version->major_(),
         (int)version->minor_(), (int)version->sub_minor());
  printf("%s Build at %s\n", fb_model->name()->c_str(),
         fb_model->build_time()->c_str());
  if (fb_model->target()) {
    printf("For %s chip ONLY\n", fb_model->target()->c_str());
  }

  // dump peak memory usage, summary static size(weight/cmdbuf) and runtime
  // (private_gmem_size+shared_gmem_size+io_mem)
  size_t total_size = 0;
  auto &sections = *fb_model->sections();
  // static
  for (auto s : sections) {
    total_size += s->size();
  }

  // runtime
  auto &programs = *fb_model->programs();
  size_t share_size = 0;
  size_t io_size = 0;
  for (auto p : programs) {
    total_size += p->neuron_size();
    total_size += p->private_gmem();
    if (share_size < p->shared_gmem()) {
      share_size = p->shared_gmem();
    }
    auto &tensor_map = *p->tensor_map();
    for (auto t : tensor_map) {
      auto gaddr = (int64_t)t->offset();
      if (gaddr != -1) {
        auto memTypeIndx = (gaddr >> 40) & 0x07;
        bool oldVersion = p->neuron_size() > 0;
        if (memTypeIndx > 1 || oldVersion) {
          if (memTypeIndx > 2) {
            // io_mem
            auto &shape = *t->shape()->dim();
            size_t type_size = dtypeSize(t->dtype());
            size_t tensor_size =
                shape[0] * shape[1] * shape[2] * shape[3] * type_size;
            io_size += tensor_size;
          }
        }
      }
    }
  }
  total_size += share_size;
  total_size += io_size;
  printf("CviModel Need ION Memory Size: (%.2f MB)\n",
         total_size / (float)(1024 * 1024));
}

void Model::dumpSections() {
  printf("\nSections:\n");
  printf("%-3s  %-10s%-25s%-12s%-12s%-s\n", "ID", "TYPE", "NAME", "SIZE",
         "OFFSET", "MD5");
  auto &sections = *fb_model->sections();
  int i = 0;
  for (auto s : sections) {
    auto type = sectionTypeToStr(s->type());
    auto name = s->name()->c_str();
    auto size = s->size();
    auto offset = s->offset();
    auto md5 = calcSectionMD5(s, s->size());
    printf("%03d  %-10s%-25s%-12d%-12d%-s\n", i++, type, name, size, offset,
           md5.c_str());
  }
}

void Model::dumpWeightMap() {
  printf("\nWeightMap:\n");
  printf("%-3s  %-10s%-10s%-8s%-4s %-4s %-4s %-4s %-s\n", "ID", "OFFSET",
         "SIZE", "TYPE", "N", "C", "H", "W", "NAME");

  auto &weights = *fb_model->weight_map();
  int i = 0;
  for (auto w : weights) {
    auto &shape = *w->shape()->dim();
    printf("%03d  %-10d%-10d%-8s%-4d %-4d %-4d %-4d %-s\n", i++,
           (int)w->offset(), w->size(), dtypeToStr(w->type()), (int)shape[0],
           (int)shape[1], (int)shape[2], (int)shape[3], w->name()->c_str());
  }
}

void Model::dumpPrograms() {
  auto &programs = *fb_model->programs();
  int idx = 0;
  for (auto p : programs) {
    auto batch_num = p->batch_num();
    auto neuron_size = p->neuron_size();
    auto private_gmem_size = p->private_gmem();
    auto shared_gmem_size = p->shared_gmem();
    auto &input_tensors = *p->input_tensors();
    auto &output_tensors = *p->output_tensors();
    printf("\nProgram #%d\n", idx++);
    printf("    %-12s: %d\n", "batch_num", batch_num);
    if (neuron_size) {
      printf("    %-12s: %d\n", "neuron_size", neuron_size);
    } else {
      printf("    %-12s: %d\n", "private_gmem_size", private_gmem_size);
      printf("    %-12s: %d\n", "shared_gmem_size", shared_gmem_size);
    }
    printf("    %-12s: ", "inputs");
    for (int i = 0; i < (int)input_tensors.size(); i++) {
      if (i != 0)
        printf(",");
      printf("%s", input_tensors[i]->c_str());
    }
    printf("\n    %-12s: ", "outputs");
    for (int i = 0; i < (int)output_tensors.size(); i++) {
      if (i != 0)
        printf(",");
      printf("%s", output_tensors[i]->c_str());
    }
    printf("\n    %-12s:\n", "routines");
    dumpRoutines(p);
    printf("\n    %-12s:\n", "tensor_map");
    // The cvimodel is old version(blow 1.1.0)
    // if neuson size is greater than 0,
    dumpTensors(p, neuron_size > 0);
  }
}

void Model::dumpTensors(const cvi::model::Program *p, bool oldVersion) {
  printf("        ");
  printf("%-3s  %-12s%-6s%-4s %-4s %-4s %-4s %-10s %-7s %-s\n", "ID", "OFFSET",
         "TYPE", "N", "C", "H", "W", "QSCALE", "MEM", "NAME");
  auto &tensors = *p->tensor_map();
  int i = 0;
  for (auto t : tensors) {
    auto &shape = *t->shape()->dim();
    std::string memType = "   -";
    auto gaddr = (int64_t)t->offset();
    if (gaddr != -1) {
      auto memTypeIndx = (gaddr >> 40) & 0x07;
      if (memTypeIndx > 1 || oldVersion) {
        if (memTypeIndx > 2) {
          memType = "io_mem";
        } else {
          memType = "private";
        }
      } else {
        memType = "shared";
      }
    }
    float qscale = t->quant() ? t->quant()->qscale() : 0;
    if (t->name()->str() == input_quanted_tensor) {
      qscale = _qscale;
    }
    printf("        ");
    if (qscale <= 0.000001 || qscale > 400.0f) {
      printf("%03d  %-12d%-6s%-4d %-4d %-4d %-4d %-10s %-7s %-s\n", i++,
             (int)t->offset(), dtypeToStr(t->dtype()), (int)shape[0],
             (int)shape[1], (int)shape[2], (int)shape[3], "-", memType.c_str(),
             t->name()->c_str());
    } else {
      printf("%03d  %-12d%-6s%-4d %-4d %-4d %-4d %-10f %-7s %-s\n", i++,
             (int)t->offset(), dtypeToStr(t->dtype()), (int)shape[0],
             (int)shape[1], (int)shape[2], (int)shape[3], qscale,
             memType.c_str(), t->name()->c_str());
    }
  }
}

bool Model::getQscaleFromDequantCpuOp(const cvi::cpu_op::Parameter *param) {
  std::string from;
  std::string to;
  float threshold = 0;
  auto &attributes = *param->attributes();
  for (auto attr : attributes) {
    if (attr->float_attr()) {
      auto _float = attr->float_attr();
      if (_float->key()->str() == "threshold") {
        threshold = _float->value();
      }
    } else if (attr->str_attr()) {
      auto _str = attr->str_attr();
      if (_str->key()->str() == "from") {
        from = _str->value()->str();
      } else if (_str->key()->str() == "to") {
        to = _str->value()->str();
      }
    }
  }
  if (threshold != 0 && from == "NONE" && to == "INT8") {
    _qscale = 128.0 / threshold;
    return true;
  }
  return false;
}

void Model::dumpRoutines(const cvi::model::Program *p) {
  auto &routines = *p->routines();
  int i = 0;
  for (auto r : routines) {
    bool tpu = r->type() == cvi::model::RoutineType_TPU;
    printf("     #%02d  %s\n", i++, tpu ? "tpu" : "cpu");
    printf("        %-8s: ", "inputs");
    int j = 0;
    for (auto name : *r->in_tensors()) {
      if (j++ != 0)
        printf(",");
      printf("%s", name->c_str());
    }
    printf("\n        %-8s: ", "outputs");
    j = 0;
    for (auto name : *r->out_tensors()) {
      if (j++ != 0)
        printf(",");
      printf("%s", name->c_str());
    }
    if (tpu) {
      std::string buf_name;
      if (r->tpu_routine()->cmdbuf_section()) {
        buf_name = r->tpu_routine()->cmdbuf_section()->str();
      } else if (r->tpu_routine()->dmabuf_section()) {
        buf_name = r->tpu_routine()->dmabuf_section()->str();
      } else {
        assert(0 && "model has not cmdbuf and dmabuf");
      }
      printf("\n        %-8s: %s\n", "section", buf_name.c_str());
    } else {
      if (r->cpu_routine()->function_section()->str() == "quant" &&
          _qscale == 0) {
        auto param = cvi::cpu_op::GetParameter(
            r->cpu_routine()->function_args()->data());
        if (getQscaleFromDequantCpuOp(param)) {
          input_quanted_tensor = (*r->out_tensors())[0]->str();
        }
      }
      printf("\n        %-8s: %s\n", "function",
             r->cpu_routine()->function_section()->c_str());
    }
  }
}

const char *Model::sectionTypeToStr(cvi::model::SectionType type) {
  switch (type) {
  case cvi::model::SectionType_WEIGHT:
    return "weight";
  case cvi::model::SectionType_CMDBUF:
    return "cmdbuf";
  case cvi::model::SectionType_DMABUF:
    return "dmabuf";
  case cvi::model::SectionType_FUNC_X86:
    return "x86_64";
  case cvi::model::SectionType_FUNC_AARCH64:
    return "aarch64";
  default:
    printf("unknown section type\n");
  }
  return "";
}

const char *Model::dtypeToStr(cvi::model::DType type) {
  switch (type) {
  case cvi::model::DType_FP32:
    return "fp32";
  case cvi::model::DType_INT32:
    return "int32";
  case cvi::model::DType_UINT32:
    return "uint32";
  case cvi::model::DType_BF16:
    return "bf16";
  case cvi::model::DType_INT16:
    return "int16";
  case cvi::model::DType_UINT16:
    return "uint16";
  case cvi::model::DType_INT8:
    return "int8";
  case cvi::model::DType_UINT8:
    return "uint8";
  default:
    printf("unknown dtype\n");
  }
  return "";
}

size_t Model::dtypeSize(cvi::model::DType type) {
  switch (type) {
  case cvi::model::DType_FP32:
    return 4;
  case cvi::model::DType_INT32:
    return 4;
  case cvi::model::DType_UINT32:
    return 4;
  case cvi::model::DType_BF16:
    return 2;
  case cvi::model::DType_INT16:
    return 2;
  case cvi::model::DType_UINT16:
    return 2;
  case cvi::model::DType_INT8:
    return 1;
  case cvi::model::DType_UINT8:
    return 1;
  default:
    printf("unknown dtype\n");
  }
  return 0;
}

static std::string getStrOfCurrentTime() {
  std::stringstream ssTime;
  auto clockNow = std::chrono::system_clock::now();
  auto t = std::chrono::system_clock::to_time_t(clockNow);
  ssTime << std::put_time(std::localtime(&t), "%Y-%m-%d %H:%M:%S");
  return ssTime.str();
}

FBTensorVector Model::cloneTensorMap(const cvi::model::Program *program) {
  std::vector<flatbuffers::Offset<cvi::model::Tensor>> tensor_vec;
  for (auto t : *program->tensor_map()) {
    auto name = t->name()->c_str();
    std::vector<int64_t> dim;
    for (auto s : *t->shape()->dim()) {
      dim.push_back(s);
    }
    auto shape = cvi::model::CreateShapeDirect(fbb, &dim);
    auto tensor =
        cvi::model::CreateTensorDirect(fbb, t->tensor_id(), name, t->offset(),
                                       t->dtype(), shape, 0, 0, t->overwrote());
    if (t->quant()) {
      auto quant = cvi::model::CreateQuantInfo(fbb, t->quant()->type(), 0, 0,
                                               t->quant()->zero_point(),
                                               t->quant()->qscale());
      std::vector<float> scale;
      if (t->scale()) {
        for (int i = 0; i < (int)t->scale()->size(); ++i) {
          scale.push_back(t->scale()->Get(i));
        }
      }

      std::vector<float> mean;
      if (t->mean()) {
        for (int i = 0; i < (int)t->mean()->size(); ++i) {
          mean.push_back(t->mean()->Get(i));
        }
      }

      std::string pixel_format;
      if (t->pixel_format()) {
        pixel_format = t->pixel_format()->str();
      }

      tensor = cvi::model::CreateTensorDirect(
          fbb, t->tensor_id(), name, t->offset(), t->dtype(), shape, 0, quant,
          t->overwrote(), scale.size() > 0 ? &scale : nullptr,
          mean.size() > 0 ? &mean : nullptr,
          pixel_format.length() > 0 ? pixel_format.c_str() : nullptr,
          t->aligned(), t->size());
    }
    tensor_vec.push_back(tensor);
  }
  return fbb.CreateVector(tensor_vec);
}

FBRoutineVector
Model::cloneRoutines(const cvi::model::Program *program, bool rename, int index,
                     std::map<std::string, std::string> *routine_name_map) {
  std::vector<flatbuffers::Offset<cvi::model::Routine>> routines;
  for (auto r : *program->routines()) {
    std::vector<flatbuffers::Offset<flatbuffers::String>> fbStrVec;
    for (auto name : *r->in_tensors()) {
      fbStrVec.push_back(fbb.CreateString(name));
    }
    auto inputs = fbb.CreateVector(fbStrVec);
    fbStrVec.clear();
    for (auto name : *r->out_tensors()) {
      fbStrVec.push_back(fbb.CreateString(name));
    }
    auto outputs = fbb.CreateVector(fbStrVec);
    if (r->type() == cvi::model::RoutineType_TPU) {
      flatbuffers::Offset<cvi::model::TpuRoutine> tpuRoutine;
      if (rename) {
        std::stringstream new_name;
        if (r->tpu_routine()->cmdbuf_section()) {
          new_name << r->tpu_routine()->cmdbuf_section()->c_str() << "_"
                   << index;
          tpuRoutine = cvi::model::CreateTpuRoutineDirect(
              fbb, new_name.str().c_str(), nullptr);
          routine_name_map->emplace(r->tpu_routine()->cmdbuf_section()->c_str(),
                                    new_name.str());
        } else {
          new_name << r->tpu_routine()->dmabuf_section()->c_str() << "_"
                   << index;
          tpuRoutine = cvi::model::CreateTpuRoutineDirect(
              fbb, nullptr, new_name.str().c_str());
          routine_name_map->emplace(r->tpu_routine()->dmabuf_section()->c_str(),
                                    new_name.str());
        }
      } else {
        const char *cmdbuf = r->tpu_routine()->cmdbuf_section()
                                 ? r->tpu_routine()->cmdbuf_section()->c_str()
                                 : nullptr;
        const char *dmabuf = r->tpu_routine()->dmabuf_section()
                                 ? r->tpu_routine()->dmabuf_section()->c_str()
                                 : nullptr;
        tpuRoutine = cvi::model::CreateTpuRoutineDirect(fbb, cmdbuf, dmabuf);
      }
      auto routine = cvi::model::CreateRoutine(fbb, cvi::model::RoutineType_TPU,
                                               inputs, outputs, tpuRoutine, 0);
      routines.push_back(routine);
    } else {
      std::vector<uint8_t> args;
      for (auto byte : *r->cpu_routine()->function_args()) {
        args.push_back(byte);
      }
      auto cpuRoutine = cvi::model::CreateCpuRoutineDirect(
          fbb, r->cpu_routine()->function_section()->c_str(), &args);
      auto routine = cvi::model::CreateRoutine(fbb, r->type(), inputs, outputs,
                                               0, cpuRoutine);
      routines.push_back(routine);
    }
  }
  return fbb.CreateVector(routines);
}

FBProgramVector Model::clonePrograms(
    std::vector<std::shared_ptr<Model>> &models,
    std::vector<std::map<std::string, std::string>> &routine_name_maps) {
  std::vector<flatbuffers::Offset<cvi::model::Program>> programs;
  routine_name_maps.clear();
  for (uint32_t i = 0; i < models.size(); ++i) {
    for (auto p : *models[i]->fb_model->programs()) {
      auto tensor_map = cloneTensorMap(p);
      std::vector<flatbuffers::Offset<flatbuffers::String>> fbStrVec;
      for (auto name : *p->input_tensors()) {
        fbStrVec.push_back(fbb.CreateString(name));
      }
      auto inputs = fbb.CreateVector(fbStrVec);
      fbStrVec.clear();
      for (auto name : *p->output_tensors()) {
        fbStrVec.push_back(fbb.CreateString(name));
      }
      auto outputs = fbb.CreateVector(fbStrVec);
      std::map<std::string, std::string> routine_name_map;
      auto routines = cloneRoutines(p, true, i, &routine_name_map);
      routine_name_maps.emplace_back(std::move(routine_name_map));
      auto program = cvi::model::CreateProgram(
          fbb, p->batch_num(), p->neuron_size(), inputs, outputs, tensor_map,
          routines, p->shared_gmem(), p->private_gmem());
      programs.push_back(program);
    }
  }
  return fbb.CreateVector(programs);
}

typedef struct {
  int id;
  const cvi::model::Section *section;
  uint32_t size;
} weight_section_t;

FBSectionVector Model::cloneSections(
    std::vector<std::shared_ptr<Model>> &models,
    std::vector<uint8_t> &sections_buf,
    std::vector<std::map<std::string, std::string>> &routine_name_maps,
    std::map<int64_t, std::vector<uint8_t>> &weight_data_map) {
  uint32_t offset = 0;
  std::string weight_md5 = "";
  std::vector<flatbuffers::Offset<cvi::model::Section>> section_vec;
  std::vector<weight_section_t> weight_sections;
  std::vector<uint8_t> weight_buffer;
  assert(models.size() == routine_name_maps.size());

  uint8_t bit_buf_type = 0;
  uint32_t weight_buf_size = 0;
  uint32_t w_idx = 0;

  for (uint32_t i = 0; i < models.size(); ++i) {
    for (auto s : *models[i]->fb_model->sections()) {
      if (s->type() == cvi::model::SectionType_WEIGHT) {
        if (s->size() > weight_buf_size) {
          weight_buf_size = s->size();
          w_idx = i;
        }
      } else if (s->type() == cvi::model::SectionType_CMDBUF) {
        bit_buf_type |= 0x01;
      } else if (s->type() == cvi::model::SectionType_DMABUF) {
        bit_buf_type |= 0x10;
      }
    }
  }

  if (bit_buf_type == 0x11) {
    printf("WARN: models can't include both dmabuf and cmdbuf!\n");
    exit(1);
  }

  weight_buffer.resize(weight_buf_size, 0);
  for (auto &weight_data : weight_data_map) {
    assert(weight_data.first + weight_data.second.size() <= weight_buf_size);
    memcpy(weight_buffer.data() + weight_data.first, weight_data.second.data(),
           weight_data.second.size());
  }

  for (uint32_t i = 0; i < models.size(); ++i) {
    for (auto s : *models[i]->fb_model->sections()) {
      std::string section_name;
      std::string md5;
      if (s->type() == cvi::model::SectionType_WEIGHT) {
        if (i != w_idx) {
          continue;
        }
        section_name = s->name()->c_str();
      } else {
        section_name = routine_name_maps[i][s->name()->c_str()];
        assert(!section_name.empty());
      }
      printf("add section, name:%s type:%d\n", section_name.c_str(),
             (int)s->type());
      auto section = cvi::model::CreateSectionDirect(
          fbb, s->type(), section_name.c_str(), s->size(), offset, s->encrypt(),
          s->compress(), s->decompressed_size());
      section_vec.push_back(section);
      if (s->type() == cvi::model::SectionType_WEIGHT) {
        sections_buf.insert(sections_buf.end(), weight_buffer.begin(),
                            weight_buffer.end());
      } else {
        std::vector<uint8_t> buf(s->size());
        models[i]->stream.read(
            buf.data(), models[i]->binary_offset + s->offset(), s->size());
        sections_buf.insert(sections_buf.end(), buf.begin(), buf.end());
      }
      offset += s->size();
    }
  }
  return fbb.CreateVector(section_vec);
}

FBWeightVector Model::cloneWeightMap(
    std::vector<std::shared_ptr<Model>> &models,
    std::map<int64_t, std::vector<uint8_t>> &weight_data_map) {
  // map < offset, tuple<md5, size, vector<name>>>
  std::map<int64_t, std::tuple<std::string, int64_t, std::vector<std::string>>>
      merged_weights;
  std::vector<flatbuffers::Offset<cvi::model::Weight>> tensor_vec;

  bool has_redundat = false;
  for (uint32_t i = 0; i < models.size(); ++i) {
    // find weight section
    uint32_t s_idx = 0;
    for (; s_idx < models[i]->fb_model->sections()->size(); ++s_idx) {
      if ((*models[i]->fb_model->sections())[s_idx]->type() ==
          cvi::model::SectionType_WEIGHT) {
        break;
      }
    }
    auto &&w_section = (*models[i]->fb_model->sections())[s_idx];

    for (auto w : *models[i]->fb_model->weight_map()) {
      int64_t w_offset = w->offset() - WEIGHT_OFFSET;
      auto w_size = w->size();
      std::vector<uint8_t> weight_data;
      models[i]->extractWeightData(w_section, w_offset, w_size, weight_data);
      auto md5 = calcMD5(weight_data, w_size);
      auto iter = merged_weights.find(w_offset);
      if (iter != merged_weights.end()) {
        // redundant weight
        has_redundat = true;
        if (w_size != std::get<1>(iter->second) ||
            md5 != std::get<0>(iter->second)) {
          std::cout << "[ERROR] Size or MD5 not equal, models cann't be merged!"
                    << std::endl;
          exit(1);
        }

        // need add to WeightMap when weight's name isn't equal, even if they
        // offset/size/md5 are equal
        // auto &names = std::get<2>(std::get<1>(*iter));
        auto &names = std::get<2>(iter->second);
        auto name_iter =
            std::find(names.begin(), names.end(), w->name()->str());
        if (name_iter == names.end()) {
          names.emplace_back(w->name()->str());
        } else {
          continue;
        }
      } else {
        std::vector<std::string> names;
        names.emplace_back(w->name()->str());
        merged_weights[w_offset] =
            std::make_tuple(md5, w_size, std::move(names));
        weight_data_map[w_offset].swap(weight_data);
      }
      std::vector<int64_t> dim;
      for (auto s : *w->shape()->dim()) {
        dim.push_back(s);
      }
      auto shape = cvi::model::CreateShapeDirect(fbb, &dim);
      auto weight = cvi::model::CreateWeightDirect(
          fbb, w->name()->c_str(), w->offset(), w->size(), shape, w->type());
      tensor_vec.push_back(weight);
    }
  }
  if (!has_redundat) {
    std::cout << "[WARNNING] No redundant weight!\n" << std::endl;
  }
  return fbb.CreateVector(tensor_vec);
}

void Model::merge(std::vector<std::shared_ptr<Model>> &models,
                  std::string &dst) {
  cvi::model::Version modelVersion = cvi::model::Version(
      cvi::model::MajorVersion_value, cvi::model::MinorVersion_value,
      cvi::model::SubMinorVersion_value);
  std::vector<std::map<std::string, std::string>> routine_name_maps;
  std::map<int64_t, std::vector<uint8_t>> weight_data_map;
  auto modelName = fbb.CreateString(fb_model->name());
  auto modelBuildTime = fbb.CreateString(getStrOfCurrentTime());
  auto modelMlirVersion = fb_model->mlir_version()
                              ? fbb.CreateString(fb_model->mlir_version())
                              : fbb.CreateString("unknown");
  auto fbTarget = fbb.CreateString(fb_model->target());
  auto modelWeight = cloneWeightMap(models, weight_data_map);
  auto modelPrograms = clonePrograms(models, routine_name_maps);
  std::vector<uint8_t> sections_buf;
  auto modelSections =
      cloneSections(models, sections_buf, routine_name_maps, weight_data_map);
  auto newModel = cvi::model::CreateModel(
      fbb, &modelVersion, modelName, modelBuildTime, 0, 0, modelWeight,
      modelPrograms, modelSections, fbTarget, modelMlirVersion);
  fbb.Finish(newModel);

  MODEL_HEADER modelHeader;
  std::string magic = u8"CviModel";
  std::string pad = u8"AA";
  memcpy(modelHeader.magic, magic.c_str(), sizeof(modelHeader.magic));
  memcpy(modelHeader.padding, pad.c_str(), sizeof(modelHeader.padding));
  memset(modelHeader.chip, 0, 16);
  memcpy(modelHeader.chip, this->header.chip, sizeof(modelHeader.chip));
  modelHeader.body_size = fbb.GetSize();
  modelHeader.major = cvi::model::MajorVersion_value;
  modelHeader.minor = cvi::model::MinorVersion_value;

  std::ofstream of(dst, std::ofstream::out | std::ofstream::binary |
                            std::ofstream::trunc);
  of.write((const char *)&modelHeader, sizeof(modelHeader));
  of.write((const char *)fbb.GetBufferPointer(), fbb.GetSize());
  of.write((const char *)sections_buf.data(), sections_buf.size());
  of.close();
  printf("store cvimodel to %s\n", dst.c_str());
}
} // namespace cvtool
