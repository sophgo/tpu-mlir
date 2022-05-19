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

#include "sophgo/Dialect/Tpu/Transforms/Passes.h"
#include "sophgo/Support/MathUtils.h"
#include "sophgo/Support/Helper/Module.h"
#include "sophgo/Support/Helper/Quant.h"
#include "sophgo/Backend/BM168x/BM168x.h"
#include "sophgo/Builder/bmodel.hpp"

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"

#include <sstream>
#include <fstream>
#include <set>

using namespace llvm;
using namespace mlir;
using namespace sophgo::backend;
using namespace sophgo::helper;
using namespace flatbuffers;
namespace sophgo {
namespace tpu {

class CodegenPass : public CodegenBase<CodegenPass> {
public:
  CodegenPass() {}
  void runOnOperation() override {
    module = getOperation();
    state = Module::getState(module);
    assert(state == Module::State::TPU_ADDRESSED);
    chip = Module::getChip(module);
    bm168x = BM168x::instance(chip);
    bm168x->init();
    // store all weight to gmem
    std::vector<top::WeightOp> weights;
    for (auto func : module.getOps<FuncOp>()) {
      func.walk([&](top::WeightOp op) {
        bm168x->value_s2d(op.output(), op.read_as_byte()->data());
        weights.push_back(op);
      });
    }
    std::vector<Value> inputs;
    std::vector<Value> outputs;
    Module::getInputsOutputs(module, inputs, outputs);

    auto coeff_addr = Module::getCoeffAddr(module);
    auto coeff_size = Module::getCoeffSize(module);
    auto neuron_addr = Module::getNeuronAddr(module);
    auto neuron_size = Module::getNeuronSize(module);
    model_gen = std::make_shared<bmodel::ModelGen>();
    // add chip name
    model_gen->AddChip(chip.str());
    auto &builder = model_gen->Builder();
    auto input_tensor = CreateTensorVector(inputs);
    auto output_tensor = CreateTensorVector(outputs);
    auto coeff_mem = CreateCoeffMem(weights, coeff_addr, coeff_size);
    std::vector<uint64_t> neuron_sizes = {(uint64_t)neuron_size};
    auto neuron_sizes_fb = builder.CreateVector(neuron_sizes);
    // codegen all subnet
    auto main_func = Module::getMainFuncOp(module);
    std::vector<Offset<bmodel::SubNet>> subnet_v;
    main_func.walk([&](func::CallOp call) {
      auto subnet = CreateSubNet(call);
      subnet_v.push_back(subnet);
    });
    auto subnets = builder.CreateVector(subnet_v);
    bmodel::NetParameterBuilder npb(builder);
    npb.add_input_tensor(input_tensor);
    npb.add_output_tensor(output_tensor);
    npb.add_ctx_addr(neuron_addr);
    npb.add_ctx_size(neuron_size);
    npb.add_ctx_sizes(neuron_sizes_fb);
    npb.add_coeff_mem(coeff_mem);
    npb.add_is_dynamic(false);
    npb.add_n_dynamic(false);
    npb.add_h_w_dynamic(false);
    // create subnet
    npb.add_sub_net(subnets);
    model_gen->AddNet(Module::getName(module).str(), npb.Finish());
    model_gen->Finish();
    std::string filename = this->model_file;
    if (filename.empty()) {
      filename = Module::getName(module).str() + "_int8_bm1684.bmodel";
    }
    model_gen->Save(filename);
    bm168x->deinit();
  }

private:
  Offset<Vector<Offset<bmodel::Shape>>>
  CreateShapeVector(const ArrayRef<int64_t> &shape);
  Offset<Vector<Offset<bmodel::Tensor>>>
  CreateTensorVector(const std::vector<Value> &values);
  Offset<bmodel::SubNet> CreateSubNet(func::CallOp call);
  Offset<Vector<Offset<bmodel::CmdGroup>>> CreateCmdGroupVector();
  Offset<bmodel::CoeffMem> CreateCoeffMem(std::vector<top::WeightOp> &coeffs,
                                          uint64_t coeff_addr,
                                          uint64_t coeff_size);
  void codegen(Operation * op);
  void codegen_for_group(tpu::GroupOp gOP);
private:
  ModuleOp module;
  StringRef state;
  StringRef chip;
  BM168x *bm168x;
  std::shared_ptr<bmodel::ModelGen> model_gen;
};

std::unique_ptr<OperationPass<ModuleOp>> createCodegenPass() {
  return std::make_unique<CodegenPass>();
}

Offset<Vector<Offset<bmodel::Shape>>>
CodegenPass::CreateShapeVector(const ArrayRef<int64_t> &shape) {
  auto &builder = model_gen->Builder();
  std::vector<Offset<bmodel::Shape>> stage_shape_v;
  std::vector<uint64_t> dims;
  for (auto dim : shape) {
    dims.push_back(dim);
  }
  auto dim_vector = builder.CreateVector(dims);
  auto shapes = bmodel::CreateShape(builder, dim_vector);
  stage_shape_v.push_back(shapes);
  return builder.CreateVector(stage_shape_v);
}

Offset<Vector<Offset<bmodel::Tensor>>>
CodegenPass::CreateTensorVector(const std::vector<Value> &values) {
  auto &builder = model_gen->Builder();
  std::vector<Offset<bmodel::Tensor>> tensor_v;
  for (auto v : values) {
    auto op = v.getDefiningOp();
    auto op_name = Module::getName(op).str();
    auto type = Module::getStorageType(v);
    auto shape = Module::getShape(v);
    auto typeBytes = type.getIntOrFloatBitWidth() / 8;
    auto data_type = BM168x::getDataType(type);
    auto gmem_stmode = STORE_MODE_1N;
    if (typeBytes == 1) {
      gmem_stmode = STORE_MODE_4N;
    } else if (typeBytes == 2) {
      gmem_stmode = STORE_MODE_2N;
    }
    // shape info
    auto name = builder.CreateString(op_name);
    auto stage_shape = CreateShapeVector(shape);
    bmodel::TensorBuilder tb(builder);
    tb.add_name(name);
    tb.add_data_type(data_type);
    tb.add_gmem_stmode(gmem_stmode);
    tb.add_shape(stage_shape);
    tb.add_mem_type(MEM_TYPE_TPU);
    float scale = 1.0f;
    if (Quant::isUniformQuantized(v)) {
      auto qtype = Quant::getUniformQuantizedType(v);
      scale = qtype.getScale();
      if (isa<top::InputOp>(v.getDefiningOp())) {
        scale = 1 / scale;
      }
      tb.add_scale(scale);
    }
    tb.add_device_addr(Module::getAddress(v));
    tb.add_size(Module::getBytes(v));
    tensor_v.push_back(tb.Finish());
  }
  return builder.CreateVector(tensor_v);
}

Offset<bmodel::CoeffMem>
CodegenPass::CreateCoeffMem(std::vector<top::WeightOp> &coeffs,
                            uint64_t coeff_addr, uint64_t coeff_size) {
  auto data_u8 = std::make_shared<std::vector<uint8_t>>(coeff_size, 0);
  uint64_t offset = 0;
  for (auto weight : coeffs) {
    auto data = weight.read_as_byte();
    memcpy(data_u8->data() + offset, data->data(), data->size());
    offset += align_up((int64_t)data->size(), BM168x::ALIGNMENT);
  }
  assert(offset == coeff_size);
  std::vector<uint8_t> sha256(bmodel::SHA256_LEN, 0);
  bmodel::CalcSha256(data_u8->data(), coeff_size, sha256.data());
  auto binary_coeff = model_gen->WriteBinary(coeff_size, data_u8->data());
  auto coeff_sha256 = model_gen->Builder().CreateVector(sha256);
  bmodel::CoeffMemBuilder cmb(model_gen->Builder());
  cmb.add_address(coeff_addr);
  cmb.add_check_code(coeff_sha256);
  cmb.add_binary_coeff(&binary_coeff);
  return cmb.Finish();
}

Offset<Vector<Offset<bmodel::CmdGroup>>> CodegenPass::CreateCmdGroupVector() {
  std::vector<Offset<bmodel::CmdGroup>> cmd_group_v;
  auto gdma_ptr = (uint8_t *)bm168x->gdma_buffer->data();
  auto bdc_ptr = (uint8_t *)bm168x->bdc_buffer->data();
  uint32_t gdma_size =
      bm168x->gdma_buffer->size() * sizeof(uint32_t);
  uint32_t bdc_size = bm168x->bdc_buffer->size() * sizeof(uint32_t);
  int bdc_offset = 0, gdma_offset = 0;
  for (int group_idx = 0; group_idx < bm168x->cmdid_groupnum;
       group_idx++) {
    auto bdc_num = bm168x->bdc_group_id[group_idx];
    auto gdma_num = bm168x->gdma_group_id[group_idx];
    bmodel::Binary binary_bdc;
    bmodel::Binary binary_gdma;
    auto bdc_len = bm168x->get_bdc_len(bdc_num, group_idx);
    if (bdc_len != 0) {
      binary_bdc = model_gen->WriteBinary(bdc_len, bdc_ptr + bdc_offset);
      bdc_offset += bdc_len;
    }
    auto gdma_len = bm168x->get_gdma_len(gdma_num, group_idx);
    if (gdma_num != 0) {
      binary_gdma = model_gen->WriteBinary(gdma_len, gdma_ptr + gdma_offset);
      gdma_offset += gdma_len;
    }
    bmodel::CmdGroupBuilder cgb(model_gen->Builder());
    cgb.add_bdc_num(bdc_num);
    cgb.add_gdma_num(gdma_num);
    cgb.add_bdc_cmd_byte(bdc_len);
    cgb.add_gdma_cmd_byte(gdma_len);
    if (bdc_num != 0) {
      cgb.add_binary_bdc(&binary_bdc);
    }
    if (gdma_num != 0) {
      cgb.add_binary_gdma(&binary_gdma);
    }
    cmd_group_v.push_back(cgb.Finish());
  }
  if (cmd_group_v.size() == 0) {
    return 0;
  }
  return model_gen->Builder().CreateVector(cmd_group_v);
}

void CodegenPass::codegen_for_group(tpu::GroupOp gOp) {
  auto nsecs = gOp.nsecs();
  auto hsecs = gOp.hsecs();
  auto &body = gOp.body().front();
  int64_t timestep = -1;
  bm168x->divide_sync_id();
  for(uint64_t nstep = 0; nstep < nsecs; nstep++) {
    for (uint64_t hstep = 0; hstep < hsecs; hstep++) {
      body.walk([&](Operation * op) {
        auto lgOp = cast<LocalGenInterface>(op);
        auto ginfo = lgOp.getGroupInfo(nstep, hstep);
        if (ginfo.timestep != timestep) {
          bm168x->merge_sync_id();
          bm168x->divide_sync_id();
          timestep = ginfo.timestep;
        }
        lgOp.codegen_local(nstep, hstep);
      });
    }
  }
  bm168x->merge_sync_id();
}

void CodegenPass::codegen(Operation * op) {
  if (auto castOp = dyn_cast<tpu::GroupOp>(op)) {
    codegen_for_group(castOp);
  } else if(Module::isOpInGroup(op)) {
    return;
  } else if (auto castOp = dyn_cast<GlobalGenInterface>(op)) {
    castOp.codegen_global();
  }
}

Offset<bmodel::SubNet> CodegenPass::CreateSubNet(func::CallOp call) {
  bm168x->before_codegen();
  auto func = Module::getFuncOp(module, call.getCallee());
  func.walk([&](Operation * op) {
    codegen(op);
  });
  bm168x->after_codegen();
  int subnet_id = func->getAttrOfType<IntegerAttr>("id").getInt();
  std::vector<Value> inputs;
  std::vector<Value> outputs;
  Module::getInputsOutputs(call, inputs, outputs);
  auto input_tensor = CreateTensorVector(inputs);
  auto output_tensor = CreateTensorVector(outputs);
  std::vector<int> next_id_v = {};
  for (auto v : call.getResults()) {
    for (auto user : v.getUsers()) {
      if (isa<func::ReturnOp>(user)) {
        next_id_v.push_back(-1);
      } else if (auto call = dyn_cast<func::CallOp>(user)) {
        auto func = Module::getFuncOp(module, call.getCallee());
        auto id = func->getAttrOfType<IntegerAttr>("id").getInt();
        next_id_v.push_back(id);
      } else {
        llvm_unreachable("next op is illegal");
      }
    }
  }
  auto &builder = model_gen->Builder();
  auto next_ids = builder.CreateVector(next_id_v);
  auto cmd_group = CreateCmdGroupVector();
  bmodel::SubNetBuilder snb(builder);
  snb.add_cmd_group(cmd_group);
  snb.add_is_dynamic(false);
  snb.add_subnet_mode(SUBNET_MODE_TPU);
  snb.add_input_tensor(input_tensor);
  snb.add_output_tensor(output_tensor);
  snb.add_id(subnet_id);
  snb.add_next_subnet_ids(next_ids);
  return snb.Finish();
}

} // namespace tpu
} // namespace sophgo
