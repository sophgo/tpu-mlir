//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
/*
 * Written by:
 *   Pengchao Hu <pengchao.hu@bitmain.com>
 * Created Time: 2018-12-07 15:34
 */

#include "tpu_mlir/Builder/bmodel.hpp"
#include <memory.h>
#include <stdlib.h>
#include <ctime>
#include <iostream>

using bmodel::Binary;
using bmodel::Model;
using bmodel::ModelCtx;
using bmodel::ModelGen;
using bmodel::NetDynamic;
using bmodel::NetStatic;
using bmodel::Tensor;
using flatbuffers::FlatBufferBuilder;
using flatbuffers::Offset;
using flatbuffers::Vector;
using std::string;
using std::vector;

const char *BMODEL_VERSION = "2.2";
const char *BMODEL_TYPE = "B";
// bmtap2 bmodel magic is 0xFF55AAFF
// net compiler bmodel magic is 0xFF55AAEE
const uint32_t BMODEL_MAGIC = 0xFF55AAEE;

#define BMODEL_LOG(severity) \
  std::cout << "[BMODEL][" << __func__ << ":" << __LINE__ << "] " << #severity << ": "

#define ASSERT(_cond)                           \
  do {                                          \
    if (!(_cond)) {                             \
      BMODEL_LOG(FATAL) << #_cond << std::endl; \
      exit(-1);                                 \
    }                                           \
  } while (0)

ModelGen::ModelGen(uint32_t reserved_size)
{
  binary_.reserve(reserved_size);
  max_neuron_size_ = 0;
}

FlatBufferBuilder &ModelGen::Builder()
{
  return builder_;
}

ModelGen::~ModelGen()
{
  builder_.ReleaseBufferPointer();
}

Binary ModelGen::WriteBinary(size_t size, uint8_t *data)
{
  // ASSERT(size != 0 && data != NULL);
  for (auto &binary : binary_vector_) {
    if (binary.size() != size) {
      continue;
    }
    if (memcmp(data, binary_.data() + binary.start(), size) == 0) {
      return binary;
    }
  }
  uint64_t start = binary_.size();
  binary_.insert(binary_.end(), size, 0);
  memcpy(binary_.data() + start, data, size);
  Binary new_bin(start, size);
  binary_vector_.push_back(new_bin);
  return new_bin;
}

void ModelGen::AddNet(const flatbuffers::Offset<bmodel::Net> &net)
{
  nets_.push_back(net);
}

void ModelGen::AddNet(string net_name, const Offset<NetParameter> &parameter, uint32_t *net_idx,
                      uint32_t *stage_idx)
{
  ASSERT(net_name.empty() == false);
  auto net_new = reinterpret_cast<const NetParameter *>(builder_.GetCurrentBufferPointer() +
                                                        builder_.GetSize() - parameter.o);
  if (net_new->ctx_size() > max_neuron_size_) {
    max_neuron_size_ = net_new->ctx_size();
  }
  uint32_t idx = 0;
  for (; idx < net_vector_.size(); idx++) {
    if (net_vector_[idx].name == net_name) {
      break;
    }
  }
  if (net_idx != NULL) {
    *net_idx = idx;
  }
  if (idx == net_vector_.size()) {  // if not found
    NET_INFO_T net_info;
    net_info.name = net_name;
    net_info.parameters.push_back(parameter);
    net_vector_.push_back(net_info);
    if (stage_idx != NULL) {
      *stage_idx = 0;
    }
  } else {  // if found
    auto &parameters = net_vector_[idx].parameters;
    for (auto &net_offset : parameters) {
      // check whether conflict
      auto net_old = reinterpret_cast<const NetParameter *>(builder_.GetCurrentBufferPointer() +
                                                            builder_.GetSize() - net_offset.o);
      if (net_old->is_dynamic() != net_new->is_dynamic()) {
        BMODEL_LOG(FATAL) << "net[" << net_name
                          << "] cannot has dynamic and static at the same time" << std::endl;
        exit(-1);
      }
      if (IsTensorConflict(net_old->input_tensor(), net_new->input_tensor())) {
        BMODEL_LOG(FATAL) << "net[" << net_name << "] input tensors is conflict" << std::endl;
        exit(-1);
      }
      if (net_old->h_w_dynamic() != net_new->h_w_dynamic() ||
          net_old->n_dynamic() != net_new->n_dynamic()) {
        BMODEL_LOG(FATAL) << "net[" << net_name << "] dynamic is conflict." << std::endl;
        exit(-1);
      }
      bool old_have_subnet =
          (net_old->sub_net() != NULL) ? (net_old->sub_net()->size() > 1) : false;
      bool new_have_subnet =
          (net_new->sub_net() != NULL) ? (net_new->sub_net()->size() > 1) : false;
      if (old_have_subnet != new_have_subnet) {
        BMODEL_LOG(FATAL) << "net[" << net_name << "] sub net is conflict." << std::endl;
        exit(-1);
      }
    }
    if (stage_idx != NULL) {
      *stage_idx = parameters.size();
    }
    parameters.push_back(parameter);
  }
}

bool ModelGen::IsShapeSame(const bmodel::Shape *left, const bmodel::Shape *right)
{
  if (left->dim()->size() != right->dim()->size()) {
    return false;
  }
  for (uint32_t index = 0; index < left->dim()->size(); index++) {
    if (left->dim()->Get(index) != right->dim()->Get(index)) {
      return false;
    }
  }
  return true;
}

bool ModelGen::IsTensorConflict(const Vector<Offset<Tensor>> *left,
                                const Vector<Offset<Tensor>> *right)
{
  if (left->size() != right->size()) {
    BMODEL_LOG(ERROR) << "tensor size is not the same, [" << left->size() << "] vs ["
                      << right->size() << "]" << std::endl;
    return true;
  }
  bool shape_same = true;
  for (uint32_t index = 0; index < left->size(); index++) {
    auto left_i = left->Get(index);
    auto right_i = right->Get(index);
    if (left_i->name()->str() != right_i->name()->str()) {
      BMODEL_LOG(ERROR) << "tensor name is not the same, [" << left_i->name() << "] vs ["
                        << right_i->name() << "]" << std::endl;
      return true;
    }
    if (left_i->data_type() != right_i->data_type()) {
      BMODEL_LOG(ERROR) << "tensor type is not the same, [" << left_i->data_type() << "] vs ["
                        << right_i->data_type() << "]" << std::endl;
      return true;
    }
    if (left_i->scale() != right_i->scale()) {
      BMODEL_LOG(ERROR) << "tensor scale is not the same, [" << left_i->scale() << "] vs ["
                        << right_i->scale() << "]" << std::endl;
      return true;
    }
    // if (left_i->gmem_stmode() != right_i->gmem_stmode()) {
    //  BMODEL_LOG(ERROR) << "tensor stmode is not the same, [" << left_i->gmem_stmode() << "] vs ["
    //  << right_i->gmem_stmode() << "]" << std::endl; return true;
    //}
    if (left_i->shape()->size() != right_i->shape()->size()) {
      BMODEL_LOG(ERROR) << "tensor shape count is not the same, [" << left_i->shape()->size()
                        << "] vs [" << right_i->shape()->size() << "]" << std::endl;
      return true;
    }
    for (uint32_t i = 0; i < left_i->shape()->size(); i++) {
      if (false == IsShapeSame(left_i->shape()->Get(i), right_i->shape()->Get(i))) {
        shape_same = false;
      }
    }
  }
  if (shape_same) {
    BMODEL_LOG(ERROR) << "tensor shape should not be the same" << std::endl;
    return true;
  }
  return false;
}

void ModelGen::AddChip(const std::string &arch_name)
{
  ASSERT(!arch_name.empty());
  chip_ = arch_name;
}

void ModelGen::Finish(const string &filename)
{
  this->Finish();
  this->Save(filename);
}

size_t ModelGen::Finish()
{
  // create net
  for (auto net_info : net_vector_) {
    Offset<Vector<Offset<NetParameter>>> parameter = 0;
    if (net_info.parameters.empty() == false) {
      parameter = builder_.CreateVector(net_info.parameters);
    }

    ASSERT(parameter.IsNull() == false);

    auto net_name = builder_.CreateString(net_info.name);
    bmodel::NetBuilder nb(builder_);
    nb.add_name(net_name);
    nb.add_parameter(parameter);
    nets_.push_back(nb.Finish());
  }
  if (nets_.empty()) {
    BMODEL_LOG(FATAL) << "there is no net" << std::endl;
    exit(-1);
  }

  // create model
  auto type = builder_.CreateString(BMODEL_TYPE);
  auto version = builder_.CreateString(BMODEL_VERSION);
  auto net = builder_.CreateVector(nets_);
  auto chip = builder_.CreateString(chip_);
  auto now = time(0);
  auto time = builder_.CreateString(ctime(&now));

  bmodel::ModelBuilder mb(builder_);
  mb.add_chip(chip);
  mb.add_type(type);
  mb.add_time(time);
  mb.add_version(version);
  mb.add_net(net);
  mb.add_neuron_size(max_neuron_size_);

  auto model = mb.Finish();
  builder_.Finish(model);

  // return size
  size_t size = sizeof(MODEL_HEADER_T) + builder_.GetSize() + binary_.size();
  return size;
}

uint8_t *ModelGen::GetBufferPointer()
{
  return builder_.GetBufferPointer();
}

void ModelGen::Save(const string &filename)
{
  ASSERT(!filename.empty());
  std::ofstream fout(filename, std::ios::out | std::ios::trunc | std::ios::binary);
  if (!fout) {
    BMODEL_LOG(FATAL) << "Save file[" << filename << "] failed." << std::endl;
    exit(-1);
  }
  MODEL_HEADER_T header;
  memset(&header, 0, sizeof(header));
  header.magic = BMODEL_MAGIC;
  header.header_size = sizeof(header);
  header.flatbuffers_size = builder_.GetSize();
  header.binary_size = binary_.size();
  fout.write((char *)&header, sizeof(header));
  fout.write((char *)builder_.GetBufferPointer(), builder_.GetSize());
  fout.write((char *)binary_.data(), binary_.size());
  fout.close();
}

void ModelGen::Save(void *buffer)
{
  ASSERT(buffer != NULL);
  MODEL_HEADER_T *p_header = (MODEL_HEADER_T *)buffer;
  memset(p_header, 0, sizeof(MODEL_HEADER_T));
  p_header->magic = BMODEL_MAGIC;
  p_header->header_size = sizeof(MODEL_HEADER_T);
  p_header->flatbuffers_size = builder_.GetSize();
  p_header->binary_size = binary_.size();
  uint8_t *p_flb = (uint8_t *)buffer + p_header->header_size;
  memcpy(p_flb, builder_.GetBufferPointer(), p_header->flatbuffers_size);
  uint8_t *p_binary = p_flb + p_header->flatbuffers_size;
  memcpy(p_binary, binary_.data(), p_header->binary_size);
}

ModelCtx::ModelCtx(const string &filename) : model_gen_(NULL), model_(NULL), bmodel_pointer_(NULL)
{
  // read file
  file_.open(filename, std::ios::binary | std::ios::in);
  if (!file_) {
    BMODEL_LOG(FATAL) << "File[" << filename << "] open failed." << std::endl;
    exit(-1);
  }
  file_.seekg(0, std::ios::end);
  size_t length = file_.tellg();
  if (length <= sizeof(header_)) {
    BMODEL_LOG(FATAL) << "File[" << filename << "] is broken ." << std::endl;
    exit(-1);
  }
  file_.seekg(0, std::ios::beg);

  // read header and check
  memset(&header_, 0, sizeof(header_));
  file_.read((char *)&header_, sizeof(header_));
  if (header_.magic != BMODEL_MAGIC) {
    BMODEL_LOG(FATAL) << "File[" << filename << "] is broken .." << std::endl;
    exit(-1);
  }
  if (length < header_.header_size + header_.flatbuffers_size + header_.binary_size) {
    BMODEL_LOG(FATAL) << "File[" << filename << "] is broken ..." << std::endl;
    exit(-1);
  }
  binary_offset_ = header_.header_size + header_.flatbuffers_size;
  model_buffer_ = (void *)malloc(header_.flatbuffers_size);
  ASSERT(model_buffer_ != NULL);
  file_.read((char *)model_buffer_, header_.flatbuffers_size);
  flatbuffers::Verifier v((uint8_t *)model_buffer_, header_.flatbuffers_size);
  if (!bmodel::VerifyModelBuffer(v)) {
    BMODEL_LOG(FATAL) << "Model file[" << filename << "] is broken." << std::endl;
    model_ = bmodel::GetModel(model_buffer_);
    if (model_ != NULL) {
      BMODEL_LOG(FATAL) << "=========== More Information ===========" << std::endl;
      BMODEL_LOG(FATAL) << "Version: " << model_->type()->c_str() << "."
                        << model_->version()->c_str() << std::endl;
      BMODEL_LOG(FATAL) << "Chip: " << model_->chip()->c_str() << std::endl;
      BMODEL_LOG(FATAL) << "Date: " << model_->time()->c_str() << std::endl;
    }
    exit(-1);
  }
  model_ = bmodel::GetModel(model_buffer_);
  ASSERT(model_ != NULL);
  update_bmodel();
}

ModelCtx::ModelCtx(const void *bmodel_data, size_t size)
    : model_gen_(NULL), model_(NULL), model_buffer_(NULL), bmodel_pointer_(NULL)
{
  ASSERT(bmodel_data != NULL);
  if (size <= sizeof(header_)) {
    BMODEL_LOG(FATAL) << "Bmodel data is broken ." << std::endl;
    exit(-1);
  }

  // read header and check
  memcpy(&header_, bmodel_data, sizeof(header_));
  if (header_.magic != BMODEL_MAGIC) {
    BMODEL_LOG(FATAL) << "Bmodel data is broken .." << std::endl;
    exit(-1);
  }
  if (size < header_.header_size + header_.flatbuffers_size + header_.binary_size) {
    BMODEL_LOG(FATAL) << "Bmodel data is broken ..." << std::endl;
    exit(-1);
  }
  binary_offset_ = header_.header_size + header_.flatbuffers_size;
  model_buffer_ = (void *)malloc(header_.flatbuffers_size);
  ASSERT(model_buffer_ != NULL);
  memcpy(model_buffer_, (uint8_t *)bmodel_data + header_.header_size, header_.flatbuffers_size);
  flatbuffers::Verifier v((uint8_t *)model_buffer_, header_.flatbuffers_size);
  if (!bmodel::VerifyModelBuffer(v)) {
    BMODEL_LOG(FATAL) << "Model data is broken ...." << std::endl;
    model_ = bmodel::GetModel(model_buffer_);
    if (model_ != NULL) {
      BMODEL_LOG(FATAL) << "=========== More Information ===========" << std::endl;
      BMODEL_LOG(FATAL) << "Version: " << model_->type()->c_str() << "."
                        << model_->version()->c_str() << std::endl;
      BMODEL_LOG(FATAL) << "Chip: " << model_->chip()->c_str() << std::endl;
      BMODEL_LOG(FATAL) << "Date: " << model_->time()->c_str() << std::endl;
    }
    exit(-1);
  }
  model_ = bmodel::GetModel(model_buffer_);
  ASSERT(model_ != NULL);
  update_bmodel();
  bmodel_pointer_ = bmodel_data;
}

ModelCtx::operator bool()
{
  return model_ != NULL;
}

const Model *ModelCtx::model()
{
  return model_;
}

ModelCtx::~ModelCtx()
{
  if (model_gen_ != NULL) {
    delete model_gen_;
  }
  if (model_buffer_ != NULL) {
    free(model_buffer_);
  }
}

const void *ModelCtx::data() const
{
  return model_buffer_;
}

const bmodel::MODEL_HEADER_T &ModelCtx::header() const
{
    return header_;
}

void ModelCtx::read_binary(const Binary *binary, uint8_t *buffer)
{
  read_binary(binary, 0, buffer, binary->size());
}

// read binary from offset
void ModelCtx::read_binary(const Binary *binary, uint32_t offset, uint8_t *buffer, uint32_t size)
{
  ASSERT(binary != NULL);
  ASSERT(buffer != NULL);
  ASSERT(size + offset <= binary->size());
  if (bmodel_pointer_ == NULL) {  // from file
    file_.seekg(binary_offset_ + binary->start() + offset, std::ios::beg);
    file_.read((char *)buffer, size);
  } else {  // from buffer
    memcpy(buffer, (uint8_t *)bmodel_pointer_ + binary_offset_ + binary->start() + offset, size);
  }
}

template <typename T>
static Offset<T> Pack(ModelGen *model_gen, const T *item)
{
  if (item == NULL) {
    return 0;
  }
  auto itemT = item->UnPack();
  auto item_offset = T::Pack(model_gen->Builder(), itemT);
  delete itemT;
  return item_offset;
}

template <typename T>
static Offset<Vector<Offset<T>>> Pack(ModelGen *model_gen, const Vector<Offset<T>> *item)
{
  if (item == NULL || item->size() == 0) {
    return 0;
  }
  vector<Offset<T>> item_v;
  for (uint32_t idx = 0; idx < item->size(); idx++) {
    item_v.push_back(Pack(model_gen, item->Get(idx)));
  }
  return model_gen->Builder().CreateVector(item_v);
}

void ModelCtx::update_net(const string &net_name, const Vector<Offset<NetStatic>> *net_static)
{
  if (net_static == NULL || net_static->size() == 0) {
    return;
  }
  for (uint32_t idx = 0; idx < net_static->size(); idx++) {
    auto net_param = net_static->Get(idx);
    auto input_offset = Pack(model_gen_, net_param->input_tensor());
    auto output_offset = Pack(model_gen_, net_param->output_tensor());
    auto cmdgroup_offset = Pack(model_gen_, net_param->cmd_group());
    auto subnet_offset = Pack(model_gen_, net_param->sub_net());
    auto coeff_offset = Pack(model_gen_, net_param->coeff_mem());

    bmodel::Binary profile;
    if (net_param->net_profile() != NULL) {
      profile = bmodel::Binary(net_param->net_profile()->start(), net_param->net_profile()->size());
    }

    bmodel::NetParameterBuilder npb(model_gen_->Builder());
    npb.add_input_tensor(input_offset);
    npb.add_output_tensor(output_offset);
    npb.add_ctx_addr(net_param->ctx_addr());
    npb.add_ctx_size(net_param->ctx_size());
    npb.add_coeff_mem(coeff_offset);
    npb.add_is_dynamic(false);
    npb.add_cmd_group(cmdgroup_offset);
    if (net_param->net_profile() != NULL) {
      npb.add_net_profile(&profile);
    }
    npb.add_sub_net(subnet_offset);
    model_gen_->AddNet(net_name, npb.Finish());
  }
}

void ModelCtx::update_net(const string &net_name, const Vector<Offset<NetDynamic>> *net_dynamic)
{
  if (net_dynamic == NULL || net_dynamic->size() == 0) {
    return;
  }
  for (uint32_t idx = 0; idx < net_dynamic->size(); idx++) {
    auto net_param = net_dynamic->Get(idx);
    auto input_offset = Pack(model_gen_, net_param->input_tensor());
    auto output_offset = Pack(model_gen_, net_param->output_tensor());
    auto stageir_offset = Pack(model_gen_, net_param->stage_ir());
    ASSERT(net_param->binary_ir() != NULL);
    bmodel::Binary binaryIR(net_param->binary_ir()->start(), net_param->binary_ir()->size());

    auto subnet_offset = Pack(model_gen_, net_param->sub_net());
    auto coeff_offset = Pack(model_gen_, net_param->coeff_mem());

    bmodel::NetParameterBuilder npb(model_gen_->Builder());
    npb.add_input_tensor(input_offset);
    npb.add_output_tensor(output_offset);
    npb.add_ctx_addr(net_param->ctx_addr());
    npb.add_ctx_size(net_param->ctx_size());
    npb.add_coeff_mem(coeff_offset);
    npb.add_is_dynamic(true);
    npb.add_n_dynamic(net_param->n_dynamic());
    npb.add_h_w_dynamic(net_param->h_w_dynamic());
    npb.add_stage_ir(stageir_offset);
    npb.add_binary_ir(&binaryIR);
    npb.add_sub_net(subnet_offset);
    model_gen_->AddNet(net_name, npb.Finish());
  }
}

void ModelCtx::update_bmodel()
{
  bool need_update = false;
  for (uint32_t net_idx = 0; net_idx < model_->net()->size(); net_idx++) {
    auto net = model_->net()->Get(net_idx);
    if (net->parameter() != NULL) {
      continue;
    }
    need_update = true;
    break;
  }
  if (need_update == false) {
    return;
  }
  model_gen_ = new bmodel::ModelGen(0);
  model_gen_->AddChip(model_->chip()->str());
  for (uint32_t net_idx = 0; net_idx < model_->net()->size(); net_idx++) {
    auto net = model_->net()->Get(net_idx);
    if (net->parameter() != NULL) {
      model_gen_->AddNet(Pack(model_gen_, net));
      continue;
    }
    update_net(net->name()->str(), net->net_static());
    update_net(net->name()->str(), net->net_dynamic());
  }
  model_gen_->Finish();
  model_ = bmodel::GetModel(model_gen_->GetBufferPointer());
}

/*********************************************************************
* Filename:   sha256.c
* Author:     Brad Conte (brad AT bradconte.com)
* Copyright:
* Disclaimer: This code is presented "as is" without any guarantees.
* Details:    Implementation of the SHA-256 hashing algorithm.
              SHA-256 is one of the three algorithms in the SHA2
              specification. The others, SHA-384 and SHA-512, are not
              offered in this implementation.
              Algorithm specification can be found here:
               * http://csrc.nist.gov/publications/fips/fips180-2/fips180-2withchangenotice.pdf
              This implementation uses little endian uint8_t order.
*********************************************************************/

/**************************** DATA TYPES ****************************/

typedef struct {
  uint8_t data[64];
  uint32_t datalen;
  uint64_t bitlen;
  uint32_t state[8];
} SHA256_CTX;

/*********************** FUNCTION DECLARATIONS **********************/
void sha256_init(SHA256_CTX *ctx);
void sha256_update(SHA256_CTX *ctx, const uint8_t data[], size_t len);
void sha256_final(SHA256_CTX *ctx, uint8_t hash[]);

/****************************** MACROS ******************************/
#define ROTLEFT(a, b) (((a) << (b)) | ((a) >> (32 - (b))))
#define ROTRIGHT(a, b) (((a) >> (b)) | ((a) << (32 - (b))))

#define CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTRIGHT(x, 2) ^ ROTRIGHT(x, 13) ^ ROTRIGHT(x, 22))
#define EP1(x) (ROTRIGHT(x, 6) ^ ROTRIGHT(x, 11) ^ ROTRIGHT(x, 25))
#define SIG0(x) (ROTRIGHT(x, 7) ^ ROTRIGHT(x, 18) ^ ((x) >> 3))
#define SIG1(x) (ROTRIGHT(x, 17) ^ ROTRIGHT(x, 19) ^ ((x) >> 10))

/**************************** VARIABLES *****************************/
static const uint32_t k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

/*********************** FUNCTION DEFINITIONS ***********************/
void sha256_transform(SHA256_CTX *ctx, const uint8_t data[])
{
  uint32_t a, b, c, d, e, f, g, h, i, j, t1, t2, m[64];

  for (i = 0, j = 0; i < 16; ++i, j += 4)
    m[i] = (data[j] << 24) | (data[j + 1] << 16) | (data[j + 2] << 8) | (data[j + 3]);
  for (; i < 64; ++i)
    m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];

  a = ctx->state[0];
  b = ctx->state[1];
  c = ctx->state[2];
  d = ctx->state[3];
  e = ctx->state[4];
  f = ctx->state[5];
  g = ctx->state[6];
  h = ctx->state[7];

  for (i = 0; i < 64; ++i) {
    t1 = h + EP1(e) + CH(e, f, g) + k[i] + m[i];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }

  ctx->state[0] += a;
  ctx->state[1] += b;
  ctx->state[2] += c;
  ctx->state[3] += d;
  ctx->state[4] += e;
  ctx->state[5] += f;
  ctx->state[6] += g;
  ctx->state[7] += h;
}

void sha256_init(SHA256_CTX *ctx)
{
  ctx->datalen = 0;
  ctx->bitlen = 0;
  ctx->state[0] = 0x6a09e667;
  ctx->state[1] = 0xbb67ae85;
  ctx->state[2] = 0x3c6ef372;
  ctx->state[3] = 0xa54ff53a;
  ctx->state[4] = 0x510e527f;
  ctx->state[5] = 0x9b05688c;
  ctx->state[6] = 0x1f83d9ab;
  ctx->state[7] = 0x5be0cd19;
}

void sha256_update(SHA256_CTX *ctx, const uint8_t data[], size_t len)
{
  uint32_t i;

  for (i = 0; i < len; ++i) {
    ctx->data[ctx->datalen] = data[i];
    ctx->datalen++;
    if (ctx->datalen == 64) {
      sha256_transform(ctx, ctx->data);
      ctx->bitlen += 512;
      ctx->datalen = 0;
    }
  }
}

void sha256_final(SHA256_CTX *ctx, uint8_t hash[])
{
  uint32_t i;

  i = ctx->datalen;

  // Pad whatever data is left in the buffer.
  if (ctx->datalen < 56) {
    ctx->data[i++] = 0x80;
    while (i < 56)
      ctx->data[i++] = 0x00;
  } else {
    ctx->data[i++] = 0x80;
    while (i < 64)
      ctx->data[i++] = 0x00;
    sha256_transform(ctx, ctx->data);
    memset(ctx->data, 0, 56);
  }

  // Append to the padding the total message's length in bits and transform.
  ctx->bitlen += ctx->datalen * 8;
  ctx->data[63] = ctx->bitlen;
  ctx->data[62] = ctx->bitlen >> 8;
  ctx->data[61] = ctx->bitlen >> 16;
  ctx->data[60] = ctx->bitlen >> 24;
  ctx->data[59] = ctx->bitlen >> 32;
  ctx->data[58] = ctx->bitlen >> 40;
  ctx->data[57] = ctx->bitlen >> 48;
  ctx->data[56] = ctx->bitlen >> 56;
  sha256_transform(ctx, ctx->data);

  // Since this implementation uses little endian uint8_t ordering and SHA uses big endian,
  // reverse all the uint8_ts when copying the final state to the output hash.
  for (i = 0; i < 4; ++i) {
    hash[i] = (ctx->state[0] >> (24 - i * 8)) & 0x000000ff;
    hash[i + 4] = (ctx->state[1] >> (24 - i * 8)) & 0x000000ff;
    hash[i + 8] = (ctx->state[2] >> (24 - i * 8)) & 0x000000ff;
    hash[i + 12] = (ctx->state[3] >> (24 - i * 8)) & 0x000000ff;
    hash[i + 16] = (ctx->state[4] >> (24 - i * 8)) & 0x000000ff;
    hash[i + 20] = (ctx->state[5] >> (24 - i * 8)) & 0x000000ff;
    hash[i + 24] = (ctx->state[6] >> (24 - i * 8)) & 0x000000ff;
    hash[i + 28] = (ctx->state[7] >> (24 - i * 8)) & 0x000000ff;
  }
}

void bmodel::CalcSha256(const uint8_t *buffer, uint64_t size, uint8_t sha256[bmodel::SHA256_LEN])
{
  SHA256_CTX ctx;
  sha256_init(&ctx);
  sha256_update(&ctx, buffer, size);
  sha256_final(&ctx, sha256);
}

static size_t get_tensor_buffer_size(const bmodel::Tensor* tensor){
  auto dims = tensor->shape()->Get(0)->dim()->size();
  // use sizeof(int) instead of the concrete data type byte size
  //   to guarantee the size is proper for any case
  auto mem_size = sizeof (int);
  for (size_t i = 0; i < dims; i++) {
    mem_size *= tensor->shape()->Get(0)->dim()->Get(i);
  }
  return mem_size;
}

bmodel::bmodel_mem_info_t ModelCtx::get_bmodel_mem_info()
{
    ASSERT(model_!= nullptr);
    bmodel_mem_info_t info;
    memset(&info, 0, sizeof(info));
    size_t load_net_num = model()->net()->size();
    uint64_t net_max_neuron_size = 0;
    for (size_t net_idx = 0; net_idx < load_net_num; net_idx++) {
      auto net_params = model()->net()->Get(net_idx)->parameter();

      assert(net_params && net_params->size() > 0);

      auto stage_num = net_params->size();
      info.neuron_mem_size= 0;
      std::set<std::vector<uint8_t>> device_check_codes;
      std::set<std::vector<uint8_t>> host_check_codes;
      uint64_t max_neuron_size = 0;
      bool multi_subnet = false;
      for(size_t stage_idx=0; stage_idx < stage_num; stage_idx++){
        auto param = net_params->Get(0);
        auto subnets = param->sub_net();
        auto num_subnets = subnets == nullptr ? 0 : subnets->size();
        uint64_t neuron_size = param->ctx_size();
        if (neuron_size > max_neuron_size) {
          max_neuron_size = neuron_size;
        }
        if(num_subnets >1) {
          multi_subnet = true;
        }

        auto device_coeff_mem = param->coeff_mem();
        if(device_coeff_mem){
            std::vector<uint8_t> device_check_code(device_coeff_mem->check_code()->begin(), device_coeff_mem->check_code()->end());
            if(device_check_codes.count(device_check_code) == 0){
              device_check_codes.insert(device_check_code);
              info.coeff_mem_size += device_coeff_mem->binary_coeff()->size();
            }
        }

        for(size_t subnet_idx=0; subnet_idx< num_subnets; subnet_idx++){
            auto subnet = subnets->Get(subnet_idx);
            if(subnet->subnet_mode() == 1) { // cpu subnet
                auto cpu_param = subnet->cpu_param()->Get(0);
                if(!cpu_param->cpu_const()) continue;
                int cpu_const_num = cpu_param->cpu_const()->size();
                for (int i = 0; i < cpu_const_num; i++) {
                  auto host_coeff_mem = cpu_param->cpu_const()->Get(i);
                  if(!host_coeff_mem) continue;
                  std::vector<uint8_t> host_check_code(host_coeff_mem->check_code()->begin(), host_coeff_mem->check_code()->end());
                  if(host_check_codes.count(host_check_code) == 0){
                    host_check_codes.insert(host_check_code);
                    info.host_coeff_mem_size += host_coeff_mem->const_data()->size();
                  }
                }
            } else if(subnet->subnet_mode() == 0) { // run on TPU static/dynamic
                if(subnet->is_dynamic()){
                    info.dynamic_ir_mem_size += subnet->ir_len()*sizeof(uint32_t);
                } else {
                    int group_num = subnet->cmd_group()->size();
                    for (int group_idx = 0; group_idx < group_num; group_idx++) {
                      auto cmd_group = subnet->cmd_group()->Get(group_idx);
                      // just for bm1684. bm1684x instructions may be of variable length
                      if(model()->chip()->str() == "BM1682"){
                        info.bd_cmd_mem_size += cmd_group->bdc_num()*(1<<8);
                        info.gdma_cmd_mem_size += cmd_group->gdma_num()*(1<<8);
                      } else if(model()->chip()->str() == "BM1684"){
                        info.bd_cmd_mem_size += cmd_group->bdc_num()*(1<<7);
                        info.gdma_cmd_mem_size += cmd_group->gdma_num()*(1<<7);
                      }
                    }
                }
            }

        }

        info.host_neuron_mem_size  += param->cpu_mem_size()*sizeof(float);

        for (size_t i = 0; i < param->input_tensor()->size(); i++) {
          auto tensor = param->input_tensor()->Get(i);
          auto tensor_buffer_size = get_tensor_buffer_size(tensor);
          if(info.middle_buffer_size < tensor_buffer_size){
              info.middle_buffer_size = tensor_buffer_size;
          }
        }
        for (size_t i = 0; i < param->output_tensor()->size(); i++) {
          auto tensor = param->output_tensor()->Get(i);
          auto tensor_buffer_size = get_tensor_buffer_size(tensor);
          if(info.middle_buffer_size < tensor_buffer_size){
              info.middle_buffer_size = tensor_buffer_size;
          }
        }
      }
      if(multi_subnet){
          info.neuron_mem_size += max_neuron_size;
      } else {
          if(net_max_neuron_size<max_neuron_size){
              net_max_neuron_size = max_neuron_size;
          }
      }
    }
    info.neuron_mem_size += net_max_neuron_size;
    return info;
}
