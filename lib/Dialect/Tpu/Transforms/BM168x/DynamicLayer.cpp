//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include <mutex>
#include <cstddef>
#include <fstream>
#include <set>
#include <sstream>
#include <vector>
#include "tpu_mlir/Dialect/Tpu/Transforms/BM168x/DynamicLayer.hpp"
#include "tpu_mlir/Backend/BM168x/BM1684X.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LayerGroupUtil.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "mlir/Support/LLVM.h"
using namespace llvm;
using namespace mlir;
using namespace tpu_mlir::backend;
using namespace std;
namespace tpu_mlir {
namespace tpu {
// global information for multisubnet
int g_tensor_id = 0;
vector<Value> net_input_tensors;
vector<Value> net_output_tensors;
map<Value, int, value_cmp> tensor_ID;
// llvm::DenseMap<Value, int> tensor_ID;

void DynCodegenInit() {
  g_tensor_id = 0;
  tensor_ID.clear();
  net_input_tensors.clear();
  net_output_tensors.clear();
}

void SetNetIO(vector<Value> &inputs, vector<Value> &outputs) {
  for (auto v : inputs)
    net_input_tensors.emplace_back(v);
  for (auto v : outputs)
    net_output_tensors.emplace_back(v);
}

vector<Value> &get_net_output() { return net_output_tensors; }
bool is_net_input(Value v) { return v.isa<BlockArgument>(); }

bool is_net_output(Value v) {
  for (auto user : v.getUsers()) {
    if (isa<ReturnOp>(user)) {
      return true;
    } else if (auto yield_op = dyn_cast<tpu::YieldOp>(user)) {
      // check if the storeOp's output is output tensor(returnOp)
      for (auto out : yield_op->getOperands()) {
        if (out == v) {
          auto parent = yield_op->getParentOp();
          if (parent != nullptr && isa<tpu::GroupOp>(parent)) {
            for (auto val : parent->getResults()) {
              if (module::getName(val).str() == module::getName(out).str()) {
                for (auto dst_op : val.getUsers()) {
                  if (isa<ReturnOp>(dst_op)) {
                    return true;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return false;
}

#if 1
int get_tensor_id(Value v) {
  if (tensor_ID.count(v))
    return tensor_ID[v];
  else {
    tensor_ID[v] = g_tensor_id++;
    return tensor_ID[v];
  }
}
#else
int get_tensor_id(Value v) {
  if (tensor_ID.find(v) != tensor_ID.end())
    return tensor_ID[v];
  else {
    tensor_ID[v] = llvm::hash_value(v.getAsOpaquePointer());
    return tensor_ID[v];
  }
}
#endif

#define write_var(var)                                                         \
  do {                                                                         \
    if (!feign)                                                                \
      reinterpret_cast<std::decay<decltype(var)>::type &>(*u8_buffer) = var;   \
    u8_buffer += sizeof(var);                                                  \
  } while (0)

template <dynamic_layer::IOType io, typename T>
size_t copy_spec_to_buffer(void *buffer, const T &spec, bool feign = false);

template <>
size_t copy_spec_to_buffer<dynamic_layer::INPUT>(
    void *buffer, const dynamic_local_tensor_spec &spec, bool feign) {
  auto u8_buffer = static_cast<uint8_t *>(buffer);

  write_var(spec.type);
  write_var(spec.id);
  write_var(spec.addr);
  if (spec.type != DYNAMIC_NEURON) {
    write_var(spec.dtype);
    write_var(spec.dims);
    for (int i = 0; i < spec.dims; ++i)
      write_var(spec.shape[i]);
  }
  return u8_buffer - static_cast<uint8_t *>(buffer);
}

template <>
size_t copy_spec_to_buffer<dynamic_layer::OUTPUT>(
    void *buffer, const dynamic_local_tensor_spec &spec, bool feign) {
  auto u8_buffer = static_cast<uint8_t *>(buffer);

  write_var(spec.type);
  write_var(spec.id);
  write_var(spec.addr);
  write_var(spec.dtype);
  write_var(spec.consume_num);

  return u8_buffer - static_cast<uint8_t *>(buffer);
}

template <>
size_t copy_spec_to_buffer<dynamic_layer::INPUT>(
    void *buffer, const dynamic_global_tensor_spec &spec, bool feign) {
  auto u8_buffer = static_cast<uint8_t *>(buffer);
  write_var(spec.type);
  write_var(spec.id);
  write_var(spec.dtype);
  write_var(spec.dims);
  size_t volume = 1;
  for (int i = 0; i < spec.dims; ++i) {
    volume *= spec.shape[i];
    write_var(spec.shape[i]);
  }
  if (spec.type == DYNAMIC_NEURON) {
    write_var(spec.is_net_io);
  } else if (spec.type != DYNAMIC_SHAPE) {
    write_var(spec.addr);
  }

  return u8_buffer - static_cast<uint8_t *>(buffer);
}

template <>
size_t copy_spec_to_buffer<dynamic_layer::OUTPUT>(
    void *buffer, const dynamic_global_tensor_spec &spec, bool feign) {
  auto u8_buffer = static_cast<uint8_t *>(buffer);

  write_var(spec.type);
  write_var(spec.id);
  write_var(spec.dtype);
  write_var(spec.addr);
  if (spec.type == DYNAMIC_NEURON) {
    write_var(spec.is_net_io);
  }

  return u8_buffer - static_cast<uint8_t *>(buffer);
}

template <dynamic_layer::IOType io, typename T>
size_t dynamic_layer::copy_tensors_to_buffer(void *buffer, const T *specs,
                                             size_t n, bool feign) {
  size_t wrote;
  auto u32_buffer = static_cast<uint32_t *>(buffer);
  if (!feign)
    *u32_buffer = n;
  auto u8_buffer = reinterpret_cast<uint8_t *>(++u32_buffer);
  for (int i = 0; i < n; ++i) {
    wrote = copy_spec_to_buffer<io>(u8_buffer, specs[i], feign);
    u8_buffer += wrote;
  }
  return u8_buffer - static_cast<uint8_t *>(buffer);
}

#define explicit_instanciate(spec_type, io)                                    \
  template size_t dynamic_layer::copy_tensors_to_buffer<dynamic_layer::io>(    \
      void *buffer, const spec_type *, size_t n, bool feign);

explicit_instanciate(dynamic_global_tensor_spec, OUTPUT)
    explicit_instanciate(dynamic_global_tensor_spec, INPUT)
        explicit_instanciate(dynamic_local_tensor_spec, INPUT)
            explicit_instanciate(dynamic_local_tensor_spec, OUTPUT)

                size_t dynamic_layer::write_global_tensor_specs(void *buffer,
                                                                bool feign) {
  auto u8_buffer = static_cast<uint8_t *>(buffer);
  auto input_specs = this->get_input_global_tensor_specs();
  u8_buffer += this->copy_tensors_to_buffer<INPUT>(
      u8_buffer, input_specs.data(), input_specs.size(), feign);
  auto output_specs = this->get_output_global_tensor_specs();
  u8_buffer += this->copy_tensors_to_buffer<OUTPUT>(
      u8_buffer, output_specs.data(), output_specs.size(), feign);
  return u8_buffer - static_cast<uint8_t *>(buffer);
}

DynamicTensorType to_dynamic_tensor_type(Value v) {
  auto op = v.getDefiningOp();
  if (op == nullptr) {
    return DYNAMIC_NEURON;
  }
  if (isa<top::WeightOp>(op)) {
    return DYNAMIC_COEFF;
  }
  const auto def_op = v.getDefiningOp();
  if (def_op->hasTrait<trait::ShapeProducer>()) {
    return DYNAMIC_SHAPE;
  }
  for (const auto use_op : v.getUsers()) {
    if (use_op->hasTrait<trait::ShapeConsumer>()) {
      return DYNAMIC_SHAPE;
    }
  }
  if (auto load_op = dyn_cast<tpu::LoadOp>(op)) {
    // TODO: LoadOp belong to weight ?
    if (module::isWeight(load_op.getInput())) {
      return DYNAMIC_COEFF;
    }
  }
  return DYNAMIC_NEURON;
}

std::vector<dynamic_global_tensor_spec>
dynamic_layer::get_input_global_tensor_specs() {
  std::vector<dynamic_global_tensor_spec> specs;
  for (auto v : op_->getOperands()) {
    if (isa_and_nonnull<top::NoneOp, tpu::BufferOp>(v.getDefiningOp()))
      continue;
    dynamic_global_tensor_spec spec = {0};
    spec.type = to_dynamic_tensor_type(v);
    spec.id = get_tensor_id(v);
    spec.is_net_io = is_net_input(v) || is_net_output(v);
    spec.dtype = BM168x::getDataType(v);
    spec.addr = module::getAddress(v);
    auto shape = module::getShape(v);
    spec.dims = shape.size();
    for (int j = 0; j < spec.dims; j++) {
      spec.shape[j] = shape[j];
    }
    spec.host_data = nullptr;
    spec.elem_num = module::getNumElements(v);
    specs.emplace_back(spec);
  }
  return specs;
}

std::vector<dynamic_global_tensor_spec>
dynamic_layer::get_output_global_tensor_specs() {
  std::vector<dynamic_global_tensor_spec> specs;
  for (auto v : op_->getResults()) {
    if (module::isNone(v))
      continue;
    dynamic_global_tensor_spec spec = {0};
    spec.type = to_dynamic_tensor_type(v);
    spec.id = get_tensor_id(v);
    spec.is_net_io = is_net_output(v);
    spec.dtype = BM168x::getDataType(v);
    spec.addr = module::getAddress(v);
    auto shape = module::getShape(v);
    spec.dims = shape.size();
    for (int j = 0; j < spec.dims; j++) {
      spec.shape[j] = shape[j];
    }
    spec.host_data = nullptr;
    spec.elem_num = module::getNumElements(v);
    specs.emplace_back(spec);
  }
  return specs;
}

std::vector<dynamic_local_tensor_spec>
dynamic_layer::get_input_local_tensor_specs() {
  std::vector<dynamic_local_tensor_spec> specs;
  for (auto v : op_->getOperands()) {
    if (isa_and_nonnull<top::NoneOp, tpu::BufferOp>(v.getDefiningOp()))
      continue;
    dynamic_local_tensor_spec spec = {0};
    spec.id = get_tensor_id(v);
    spec.type = to_dynamic_tensor_type(v);
    spec.dtype = BM168x::getDataType(v);
    auto gi = DynLocalGenInterface::DynGetGroupInfo(v);
    spec.addr = gi.out_addr;
    auto shape = module::getShape(v);
    spec.dims = shape.size();
    for (int j = 0; j < spec.dims; j++) {
      spec.shape[j] = shape[j];
    }
    spec.host_data = nullptr;
    spec.elem_num = module::getNumElements(v);
    specs.emplace_back(spec);
  }
  return specs;
}

std::vector<dynamic_local_tensor_spec>
dynamic_layer::get_output_local_tensor_specs() {
  std::vector<dynamic_local_tensor_spec> specs;
  for (auto v : op_->getResults()) {
    if (module::isNone(v))
      continue;
    dynamic_local_tensor_spec spec = {0};
    spec.id = get_tensor_id(v);
    spec.type = to_dynamic_tensor_type(v);
    spec.dtype = BM168x::getDataType(v);
    auto shape = module::getShape(v);
    spec.dims = shape.size();
    for (int j = 0; j < spec.dims; j++) {
      spec.shape[j] = shape[j];
    }
    auto gi = DynLocalGenInterface::DynGetGroupInfo(v);
    spec.addr = gi.out_addr;
    spec.host_data = nullptr;
    spec.elem_num = module::getNumElements(v);
    specs.emplace_back(spec);
  }

  return specs;
}

size_t dynamic_layer::write_local_tensor_specs(
    void *buffer, const std::map<int, int> &consume_table, bool feign) {
  auto u8_buffer = static_cast<uint8_t *>(buffer);
  auto input_specs = this->get_input_local_tensor_specs();
  u8_buffer += this->copy_tensors_to_buffer<INPUT>(
      u8_buffer, input_specs.data(), input_specs.size(), feign);
  auto output_specs = this->get_output_local_tensor_specs();
  if (!feign) {
    for (auto &spec : output_specs) {
      auto it = consume_table.find(spec.id);
      if (it != consume_table.end()) {
        spec.consume_num = it->second;
      }
    }
  }
  u8_buffer += this->copy_tensors_to_buffer<OUTPUT>(
      u8_buffer, output_specs.data(), output_specs.size(), feign);
  return u8_buffer - static_cast<uint8_t *>(buffer);
}

size_t dynamic_layer::get_global_ir_length() {
  size_t overhead = sizeof(uint32_t) * 2; // Magic + Len
  return overhead + this->write_global_ir_impl(nullptr, true);
}

size_t dynamic_layer::get_local_ir_length() {
  size_t overhead = sizeof(uint32_t) * 2; // Magic + Len
  return overhead + this->write_local_ir_impl(nullptr, {}, true);
}

size_t dynamic_layer::write_local_ir_impl(
    void *buffer, const std::map<int, int> &consume_table, bool feign) {
  auto tensor_info_size =
      write_local_tensor_specs(buffer, consume_table, feign);

  if (auto castOp = dyn_cast<DynLocalGenInterface>(op_)) {
    return tensor_info_size +
           (size_t)castOp.dyn_codegen_local_bm1684x(
               feign ? nullptr
                     : (static_cast<char *>(buffer) + tensor_info_size));
  } else {
    assert(0);
  }
}

size_t dynamic_layer::write_global_ir_impl(void *buffer, bool feign) {
  auto tensor_info_size = write_global_tensor_specs(buffer, feign);
  if (auto castOp = dyn_cast<DynGlobalGenInterface>(op_)) {
    return tensor_info_size +
           (size_t)castOp.dyn_codegen_global_bm1684x(
               feign ? nullptr
                     : (static_cast<char *>(buffer) + tensor_info_size));
  } else {
    assert(0);
  }
}

const uint32_t Magic = 0xf00ffff;

size_t dynamic_layer::write_local_ir(void *buffer,
                                     const std::map<int, int> &consume_table) {
  auto buffer_u32 = static_cast<uint32_t *>(buffer);
  *buffer_u32 = Magic; // For sanity check
  *buffer_u32++ |= this->local_ir_version();
  auto len = this->write_local_ir_impl(buffer_u32 + 1, consume_table);
  *buffer_u32 = len;
  return len + 2 * sizeof(uint32_t);
}

size_t dynamic_layer::write_global_ir(void *buffer) {
  auto buffer_u32 = static_cast<uint32_t *>(buffer);
  *buffer_u32 = Magic; // For sanity check
  *buffer_u32++ |= this->global_ir_version();
  auto len = this->write_global_ir_impl(buffer_u32 + 1);
  *buffer_u32 = len;
  return len + 2 * sizeof(uint32_t);
}

} // namespace tpu
} // namespace tpu_mlir
