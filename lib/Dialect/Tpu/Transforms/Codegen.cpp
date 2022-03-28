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
#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "sophgo/Support/MathUtils.h"
#include "sophgo/Support/Helper/Module.h"
#include "sophgo/Support/Helper/Quant.h"
#include "sophgo/Backend/BM1684.h"
#include "sophgo/Builder/bmodel.hpp"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
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
static Offset<Vector<Offset<bmodel::Shape>>>
CreateShapeVector(FlatBufferBuilder &builder, const ArrayRef<int64_t> &shape) {
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

static Offset<Vector<Offset<bmodel::Tensor>>>
CreateTensorVector(FlatBufferBuilder &builder,
                   const std::vector<Value> &values) {
  // input tensor
  std::vector<Offset<bmodel::Tensor>> tensor_v;
  for (auto v : values) {
    auto op = v.getDefiningOp();
    auto op_name = Module::getName(op).str();
    auto name = builder.CreateString(op_name);
    auto type = Module::getType(v);
    auto shape = Module::getShape(v);
    auto typeBytes = type.getIntOrFloatBitWidth() / 8;
    auto data_type = BM1684::getType(type);
    auto gmem_stmode = STORE_MODE_1N;
    if (typeBytes == 1) {
      gmem_stmode = STORE_MODE_4N;
    } else if (typeBytes == 2) {
      gmem_stmode = STORE_MODE_2N;
    }
    // shape info
    auto stage_shape = CreateShapeVector(builder, shape);

    bmodel::TensorBuilder tb(builder);
    tb.add_name(name);
    tb.add_data_type(data_type);
    tb.add_gmem_stmode(gmem_stmode);
    tb.add_shape(stage_shape);
    float scale = 1.0f;
    if (Quant::isUniformQuantized(v)) {
      auto qtype = Quant::getQuantizedType<quant::UniformQuantizedType>(v);
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

static Offset<bmodel::CoeffMem>
CreateCoeffMem(std::shared_ptr<bmodel::ModelGen> model_gen,
               std::vector<top::WeightOp> &coeffs, uint64_t coeff_addr,
               uint64_t coeff_size) {
  auto data_u8 = std::make_shared<std::vector<uint8_t>>(coeff_size, 0);
  uint64_t offset = 0;
  for (auto weight : coeffs) {
    auto data = weight.read_as_byte();
    memcpy(data_u8->data() + offset, data->data(), data->size());
    offset += ALIGN(data->size(), BM1684::ALIGNMENT);
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

static Offset<Vector<Offset<bmodel::CmdGroup>>>
CreateCmdGroupVector(std::shared_ptr<bmodel::ModelGen> model_gen) {
  std::vector<Offset<bmodel::CmdGroup>> cmd_group_v;
  auto gdma_ptr = (uint8_t *)BM1684::instance().gdma_buffer->data();
  auto bdc_ptr = (uint8_t *)BM1684::instance().bdc_buffer->data();
  uint32_t gdma_size =
      BM1684::instance().gdma_buffer->size() * sizeof(uint32_t);
  uint32_t bdc_size = BM1684::instance().bdc_buffer->size() * sizeof(uint32_t);
  int bdc_offset = 0, gdma_offset = 0;
  for (int group_idx = 0; group_idx < BM1684::instance().cmdid_groupnum;
       group_idx++) {
    auto bdc_num = BM1684::instance().bdc_group_id[group_idx];
    auto gdma_num = BM1684::instance().gdma_group_id[group_idx];
    uint32_t bdc_len = 0, gdma_len = 0;
    uint32_t size = 0;
    bmodel::Binary binary_bdc;
    bmodel::Binary binary_gdma;
    if (bdc_num != 0) {
      size = bdc_num * BM1684::BDC_CMD_ALIGNED_NUM * sizeof(uint32_t);
      binary_bdc = model_gen->WriteBinary(size, bdc_ptr + bdc_offset);
      bdc_offset += size;
    }
    if (gdma_num != 0) {
      size = gdma_num * BM1684::GDMA_CMD_ALIGNED_NUM * sizeof(uint32_t);
      assert(gdma_offset + size <= gdma_size);
      binary_gdma = model_gen->WriteBinary(size, gdma_ptr + gdma_offset);
      gdma_offset += size;
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

class CodegenPass : public CodegenBase<CodegenPass> {
public:
  CodegenPass() {}
  void runOnOperation() override {
    auto module = getOperation();
    auto state = Module::getState(module);
    if (state != Module::State::TPU_ADDRESSED) {
      llvm_unreachable("module should be addressed");
    }
    auto chip = Module::getChip(module);
    BM1684::instance().init();
    for (auto func : module.getOps<FuncOp>()) {
      func.walk([&](top::WeightOp op) {
        // store data to gmem
        BM1684::instance().value_s2d(op.output(), op.read_as_byte()->data());
      });
    }
    for (auto func : module.getOps<FuncOp>()) {
      func.walk([&](CodegenInterface op) {
        if (chip == Module::Chip::BM1684) {
          op.codegen_int8_bm1684();
        } else if (chip == Module::Chip::BM1686) {
          op.codegen_int8_bm1686();
        }
      });
    }
    std::vector<Value> inputs;
    std::vector<Value> outputs;
    std::vector<top::WeightOp> weights;
    for (auto func : module.getOps<FuncOp>()) {
      func.walk([&](top::InputOp op) { inputs.push_back(op.output()); });
      func.walk([&](func::ReturnOp op) {
        for (auto output : op.getOperands()) {
          outputs.push_back(output);
        }
      });
      func.walk([&](top::WeightOp op) { weights.push_back(op); });
    }
    auto coeff_addr = Module::getCoeffAddr(module);
    auto coeff_size = Module::getCoeffSize(module);
    auto neuron_addr = Module::getNeuronAddr(module);
    auto neuron_size = Module::getNeuronSize(module);
    auto model_gen = std::make_shared<bmodel::ModelGen>();
    // add chip name
    model_gen->AddChip(chip.str());
    auto &builder = model_gen->Builder();
    auto input_tensor = CreateTensorVector(builder, inputs);
    auto output_tensor = CreateTensorVector(builder, outputs);
    auto coeff_mem = CreateCoeffMem(model_gen, weights, coeff_addr, coeff_size);
    auto cmd_group = CreateCmdGroupVector(model_gen);
    std::vector<uint64_t> neuron_sizes = {(uint64_t)neuron_size};
    auto neuron_sizes_fb = model_gen->Builder().CreateVector(neuron_sizes);
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
    npb.add_cmd_group(cmd_group);
    model_gen->AddNet(Module::getName(module).str(), npb.Finish());
    model_gen->Finish();
    std::string filename =
        Module::getName(module).str() + "_int8_bm1684.bmodel";
    model_gen->Save(filename);
    BM1684::instance().deinit();
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createCodegenPass() {
  return std::make_unique<CodegenPass>();
}
} // namespace tpu
} // namespace sophgo
