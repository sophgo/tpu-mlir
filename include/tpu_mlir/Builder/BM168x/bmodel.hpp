//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef LIBBMODEL_HPP_
#define LIBBMODEL_HPP_

#include <stdint.h>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include "tpu_mlir/Builder/BM168x/bmodel_generated.h"

namespace bmodel {
#ifdef __linux__
typedef struct {
  uint32_t magic;
  uint32_t header_size;
  uint32_t flatbuffers_size;
  uint32_t binary_size;
  uint32_t reserved[12];
} __attribute__((packed)) MODEL_HEADER_T;
#else
#pragma pack(push, 1)
typedef struct {
  uint32_t magic;
  uint32_t header_size;
  uint32_t flatbuffers_size;
  uint32_t binary_size;
  uint32_t reserved[12];
} MODEL_HEADER_T;
#pragma pack(pop)
#endif

typedef struct {
   uint64_t bd_cmd_mem_size;       // bd instruction total size
   uint64_t gdma_cmd_mem_size;     // gdma instruction total size
   uint64_t dynamic_ir_mem_size;   // dynamic ir total size
   uint64_t neuron_mem_size;       // total neuron mem
   uint64_t coeff_mem_size;        // total coeff size
   uint64_t middle_buffer_size;    // max input and output byte size
   uint64_t host_neuron_mem_size;  // total mem size for cpu layer IO on host
   uint64_t host_coeff_mem_size;   // total mem size for cpu layer coeff on host
} bmodel_mem_info_t;

class ModelGen {
 public:
  ModelGen(uint32_t reserved_size = 0x1000000);
  virtual ~ModelGen();
  flatbuffers::FlatBufferBuilder &Builder();
  Binary WriteBinary(size_t size, uint8_t *data);

  // add model elements
  void AddChip(const std::string &arch_name);
  void AddNet(const flatbuffers::Offset<Net> &net);
  void AddNet(std::string net_name, const flatbuffers::Offset<NetParameter> &parameter,
              uint32_t *net_idx = NULL, uint32_t *stage_idx = NULL);
  // firmware_core.so save into bmodel
  void AddKernelModule(std::string &filename, Binary &tpu_module);
  // finish and save to file
  void Finish(const std::string &filename);

  // finish and return size, but no save
  size_t Finish();
  void Save(const std::string &filename);  // save to file
  void Save(void *buffer);                 // save to buffer
  uint8_t *GetBufferPointer();

 private:
  bool IsTensorConflict(const flatbuffers::Vector<flatbuffers::Offset<Tensor>> *,
                        const flatbuffers::Vector<flatbuffers::Offset<Tensor>> *);
  bool IsShapeSame(const Shape *, const Shape *);

  typedef struct {
    std::string name;
    std::vector<flatbuffers::Offset<NetParameter>> parameters;
  } NET_INFO_T;

  typedef struct {
    std::string file_name;
    Binary binary;
  } KERNEL_MODULE_T;

  std::string chip_;
  flatbuffers::FlatBufferBuilder builder_;
  std::vector<uint8_t> binary_;
  std::vector<Binary> binary_vector_;
  std::vector<NET_INFO_T> net_vector_;
  std::vector<flatbuffers::Offset<bmodel::Net>> nets_;
  uint64_t max_neuron_size_;
  // Binary tpu_module_;
  KERNEL_MODULE_T kernel_module_;
};

class ModelCtx {
 public:
  ModelCtx(const std::string &filename);
  ModelCtx(const void *bmodel_data, size_t size);
  virtual ~ModelCtx();
  operator bool();

  const Model *model();
  // read binary data to buffer
  void read_binary(const bmodel::Binary *binary, uint8_t *buffer);
  // read binary from offset
  void read_binary(const bmodel::Binary *binary, uint64_t offset, uint8_t *buffer, uint64_t size);

  // model buffer data for parse
  const void *data() const;

  const MODEL_HEADER_T &header() const;

  bmodel_mem_info_t get_bmodel_mem_info();
 protected:
  void update_bmodel();
  void update_net(const std::string &net_name,
                  const flatbuffers::Vector<flatbuffers::Offset<NetStatic>> *net_static);
  void update_net(const std::string &net_name,
                  const flatbuffers::Vector<flatbuffers::Offset<NetDynamic>> *net_dynamic);

 private:
  MODEL_HEADER_T header_;
  ModelGen *model_gen_;
  const Model *model_;
  void *model_buffer_;
  uint32_t binary_offset_;
  std::ifstream file_;          // bmodel in file
  const void *bmodel_pointer_;  // bmodel in buffer
};

}  // namespace bmodel

#endif  // LIBBMODEL_HPP_
