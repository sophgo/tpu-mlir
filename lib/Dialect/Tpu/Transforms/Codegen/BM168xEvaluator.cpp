#include "tpu_mlir/Support/Float8.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LayerGroupDefs.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/SwPipeline.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/BM168xEvaluator.h"

#define DEBUG_TYPE "tpu_evaluator"

namespace tpu_mlir {
namespace tpu {

BM168xEvaluator::BM168xEvaluator(ModuleOp m) : module(m) {
  module::init(m);
  if (!module::isState(module::State::TPU_ADDRESSED)) {
    llvm_unreachable("mlir state not support");
  }
  backend::Arch::init(0);
  bm168x = BM168x::instance();
  auto _input_names = *module::getInputs();
  for (auto name: _input_names) {
    input_names.push_back(name.str());
  }
  auto _output_names = *module::getOutputs();
  for (auto name: _output_names) {
    output_names.push_back(name.str());
  }
  module = module::getAllModules()->at(0);
}

static inline uint64_t get_staging_bytes(Value v) {
  if (module::getStorageType(v).isInteger(4))
    return module::getNumElements(v);
  else
    return (uint64_t)module::getBytes(v);
}

void BM168xEvaluator::allocate_resources() {
  const uint64_t coeff_addr = module::getCoeffAddr(module);
  const uint64_t io_addr = module::getIOAddr(module);
  const uint64_t neuron_addr = module::getNeuronAddr(module);
  const uint64_t coeff_size = module::getCoeffSize(module);
  const uint64_t io_size = module::getIOSize(module);
  const uint64_t neuron_size = module::getNeuronSize(module);
  // see spec/include/memmap.h
  uint64_t gmem_start_addr = bm168x->GMEM_START_ADDR;
  uint64_t cmodel_gmem_start_addr = bm168x->get_cmodel_gmem_start_addr();
  if (gmem_start_addr != cmodel_gmem_start_addr) {
    // in this case front end gmem_start_addr acts as a tag
    gmem_start_addr |= cmodel_gmem_start_addr;
  }
  auto fixAddr = [&](uint64_t addr) -> uint64_t {
    if (coeff_addr <= addr && addr < coeff_addr + coeff_size) {
      return (addr - coeff_addr) + gmem_start_addr;
    } else if (io_addr <= addr && addr < io_addr + io_size) {
      return (addr - io_addr + coeff_size) + gmem_start_addr;
    } else if (neuron_addr <= addr && addr < neuron_addr + neuron_size) {
      return (addr - neuron_addr + coeff_size + io_size) + gmem_start_addr;
    }
    return addr;
  };
  auto fixValueAddr = [&](Value v) -> uint64_t {
    auto addr = fixAddr(module::getAddress(v));
    module::setAddress(v, addr);
    return addr;
  };

  all_tensor_names.clear();
  value_map.clear();
  mem_map.clear();
  module.walk<WalkOrder::PreOrder>([&](func::FuncOp func) {
    if (func.getName().str() == "main") {
    } else if (auto call = module::getCallOp(func)) {
      func.walk<WalkOrder::PreOrder>([&](Operation *op) {
        if (op == func.getOperation() || isa<top::NoneOp>(op)) {
        } else if (auto wOp = dyn_cast<top::WeightOp>(op)) {
          auto v = wOp.getOutput();
          auto addr = fixValueAddr(v);
          auto name = module::getName(v).str();
          all_weight_names.push_back(name);
          const auto data = wOp.read_as_byte();
          void* ptr = bm168x->get_system_mem_ptr(addr);
          memcpy(ptr, data->data(), data->size());
        } else {
          for (auto v : op->getOperands()) {
            if (v.getDefiningOp() == nullptr) { // v is an argument of FuncOp
              fixValueAddr(v);
              auto name = module::getName(v).str();
              value_map[name] = v;
            }
          }
          for (auto v : op->getResults()) {
            if (module::getNumElements(v) == 0)
              continue;
            fixValueAddr(v);
            auto name = module::getName(v).str();
            all_tensor_names.push_back(name);
            value_map[name] = v;
            auto bytes = get_staging_bytes(v);
            mem_map[name] = std::make_shared<staging_mem_t>(bytes);
          }
          if (auto gOp = dyn_cast<GroupOp>(op)) {
            auto &body = gOp.getBody().front();
            body.walk([&](Operation *op) {
              if (isa<tpu::LoadOp, tpu::StoreOp>(op)) {
                ;
              } else if (auto yOp = dyn_cast<tpu::YieldOp>(op)) {
                for (auto v : op->getOperands()) {
                  if (module::getNumElements(v) == 0)
                    continue;
                  fixValueAddr(v);
                  auto name = module::getName(v).str();
                  all_tensor_names.push_back(name);
                  value_map[name] = v;
                  auto bytes = get_staging_bytes(v);
                  mem_map[name] = std::make_shared<staging_mem_t>(bytes);
                }
              } else {
                for (auto v : op->getOperands()) {
                  if (module::getNumElements(v) == 0)
                    continue;
                  auto name = module::getName(v).str();
                  all_tensor_names.push_back(name);
                  value_map[name] = v;
                  auto bytes = get_staging_bytes(v);
                  mem_map[name] = std::make_shared<staging_mem_t>(bytes);
                }
              }
            });
            return WalkResult::skip();
          }
        }
        return WalkResult::advance();
      });
    }
  });

  module::detachWeightFile(); // free weight mem
}

void BM168xEvaluator::setTensor(const std::string &name, const void *data,
                                size_t size, bool is_integer) {
  auto it = value_map.find(name);
  if (it == value_map.end()) {
    llvm::errs() << "Can't find op name: " << name << "\n";
    llvm_unreachable("Error, setInput failed");
  }
  auto value = it->second;
  const auto addr = module::getAddress(value);
  size_t count = module::getNumElements(value);
  void* mem_ptr = bm168x->get_system_mem_ptr(addr);
  if (!is_integer) {
    float *p = (float *)data;
    if (module::isUniformQuantized(value)) {
      auto qtype = module::getUniformQuantizedType(value);
      for (auto i = 0; i < count; i++) {
        float d = p[i] * (float)(1 / qtype.getScale()) + (float)qtype.getZeroPoint();
        ((uint8_t*)mem_ptr)[i] = qtype.isSigned() ? to_int8(d) : to_uint8(d);
      }
    } else if (module::isCalibratedType(value)) {
      auto type = module::getStorageType(value);
      if (type.isF16()) {
        for (auto i = 0; i < count; ++i) {
          ((uint16_t*)mem_ptr)[i] = f32_to_f16(p[i]);
        }
      } else if (type.isBF16()) {
        for (auto i = 0; i < count; ++i) {
          ((uint16_t*)mem_ptr)[i] = f32_to_bf16(p[i]);
        }
      } else if (type.isFloat8E4M3FN()) {
        double scale = module::getCalibratedType(value).getMax() / get_f8e4m3_max();
        for (auto i = 0; i < count; ++i) {
          ((uint8_t*)mem_ptr)[i] = f32_to_f8e4m3(p[i] / scale, true);
        }
      } else if (type.isFloat8E5M2()) {
        for (auto i = 0; i < count; ++i) {
          ((uint8_t*)mem_ptr)[i] = f32_to_f8e4m3(p[i], true);
        }
      } else {
        llvm_unreachable("Unknown type");
      }
    } else {
      auto type = module::getStorageType(value);
      if (type.isF32() || type.isInteger(8) || type.isInteger(16) || type.isInteger(32)) {
        const auto type_size = module::getDtypeSize(value);
        memcpy(mem_ptr, data, count * type_size);
      } else if (type.isF16()) {
        for (auto i = 0; i < count; ++i) {
          ((uint16_t*)mem_ptr)[i] = f32_to_f16(p[i]);
        }
      } else if (type.isBF16()) {
        for (auto i = 0; i < count; ++i) {
          ((uint16_t*)mem_ptr)[i] = f32_to_bf16(p[i]);
        }
      } else if (type.isFloat8E4M3FN()) {
        double scale = module::getCalibratedType(value).getMax() / get_f8e4m3_max();
        for (auto i = 0; i < count; ++i) {
          ((uint8_t*)mem_ptr)[i] = f32_to_f8e4m3(p[i] / scale, true);
        }
      } else if (type.isFloat8E5M2()) {
        for (auto i = 0; i < count; ++i) {
          ((uint8_t*)mem_ptr)[i] = f32_to_f8e4m3(p[i], true);
        }
      } else {
        llvm_unreachable("Unknown type");
      }
    }
  } else {
    memcpy(mem_ptr, data, size);
  }
}

std::shared_ptr<std::vector<float>>
BM168xEvaluator::getTensor(const std::string &name) {
  auto it = mem_map.find(name);
  if (it == mem_map.end() || mem_map[name].use_count() == 0) {
    llvm::errs() << "Can't find op name: " << name << "\n";
    llvm_unreachable("Error, getTensor failed");
  }
  auto value = value_map.at(name);
  size_t count = module::getNumElements(value);
  auto mem = mem_map.at(name);
  auto data_fp32 = std::make_shared<std::vector<float>>(count);
  if (module::isUniformQuantized(value)) {
    auto qtype = module::getUniformQuantizedType(value);
    assert(module::getStorageType(value).isInteger(8));
    if (qtype.isSigned()) {
      for (auto i = 0; i < count; i++) {
        data_fp32->at(i) =
            ((int8_t)mem->at(i) - (float)qtype.getZeroPoint()) * (float)qtype.getScale();
      }
    } else {
      for (auto i = 0; i < count; i++) {
        data_fp32->at(i) =
            ((uint8_t)mem->at(i) - (float)qtype.getZeroPoint()) * (float)qtype.getScale();
      }
    }
    return std::move(data_fp32);
  } else if (module::isCalibratedType(value)) {
    auto type = module::getStorageType(value);
    void* mem_ptr = mem->data();
    if (type.isF16()) {
      for (auto i = 0; i < count; ++i) {
        data_fp32->at(i) = f16_to_f32(((uint16_t*)mem_ptr)[i]);
      }
    } else if (type.isBF16()) {
      for (auto i = 0; i < count; ++i) {
        data_fp32->at(i) = bf16_to_f32(((uint16_t*)mem_ptr)[i]);
      }
    } else if (type.isFloat8E4M3FN()) {
      double scale = module::getCalibratedType(value).getMax() / get_f8e4m3_max();
      for (auto i = 0; i < count; i++) {
        data_fp32->at(i) = f8e4m3_to_f32(((uint8_t*)mem_ptr)[i]) * scale;
      }
    } else if (type.isFloat8E5M2()) {
      for (auto i = 0; i < count; i++) {
        data_fp32->at(i) = f8e5m2_to_f32(((uint8_t*)mem_ptr)[i]);
      }
    } else {
      llvm_unreachable("Unknown type");
    }
  } else {
    auto type = module::getStorageType(value);
    void* mem_ptr = mem->data();
    if (type.isF32()) {
      memcpy(data_fp32->data(), mem_ptr, count * sizeof(float));
    } else if (type.isInteger(32)) {
      for (auto i = 0; i < count; ++i) {
        data_fp32->at(i) = ((uint32_t*)mem_ptr)[i];
      }
    } else if (type.isInteger(16)) {
      for (auto i = 0; i < count; ++i) {
        data_fp32->at(i) = ((uint16_t*)mem_ptr)[i];
      }
    } else if (type.isInteger(8) || type.isInteger(4)) {
      for (auto i = 0; i < count; ++i) {
        data_fp32->at(i) = ((uint8_t*)mem_ptr)[i];
      }
    } else if (type.isF16()) {
      for (auto i = 0; i < count; ++i) {
        data_fp32->at(i) = f16_to_f32(((uint16_t*)mem_ptr)[i]);
      }
    } else if (type.isBF16()) {
      for (auto i = 0; i < count; ++i) {
        data_fp32->at(i) = bf16_to_f32(((uint16_t*)mem_ptr)[i]);
      }
    } else if (type.isFloat8E4M3FN()) {
      double scale = module::getCalibratedType(value).getMax() / get_f8e4m3_max();
      for (auto i = 0; i < count; i++) {
        data_fp32->at(i) = f8e4m3_to_f32(((uint8_t*)mem_ptr)[i]) * scale;
      }
    } else if (type.isFloat8E5M2()) {
      for (auto i = 0; i < count; i++) {
        data_fp32->at(i) = f8e5m2_to_f32(((uint8_t*)mem_ptr)[i]);
      }
    } else {
      llvm_unreachable("Unknown type");
    }
  }
  return data_fp32;
}

llvm::ArrayRef<int64_t>
BM168xEvaluator::getTensorShape(const std::string &name) {
  auto it = value_map.find(name);
  if (it == value_map.end()) {
    llvm::errs() << "Can't find op name: " << name << "\n";
    llvm_unreachable("Error, getTensorShape failed");
  }
  return it->second.getType().cast<RankedTensorType>().getShape();
}

void BM168xEvaluator::invoke() {
  bm168x->enter_runtime();

  module.walk<WalkOrder::PreOrder>([&](func::FuncOp func) {
    if (func.getName().str() == "main") {
      return WalkResult::advance();
    }
    if (auto call = module::getCallOp(func)) {
      auto mode = getRunMode(func);

      switch (mode) {
      case RunMode::TPU_STATIC: {
        CreateSubNet(call);
      } break;
      // case RunMode::TPU_DYNAMIC: {
      //   auto subnet_ir_ = std::make_unique<SubnetIr>(dynamic_mode);
      //   CreateSubNet(call, std::move(subnet_ir_), context);
      // } break;
      // case RunMode::CPU: {
      //   CreateCPUSubNet(call);
      // } break;
      // // actually use switch subnet
      // case RunMode::LOOP:
      // case RunMode::SWITCH: {
      //   CreateSwitchSubNet(call);
      // } break;
      // case RunMode::MERGE: {
      //   CreateMergeSubNet(call);
      // } break;
      default:
        llvm_unreachable("Not Implemented");
        break;
      }
    }
    return WalkResult::advance();
  });

  bm168x->exit_runtime();
}

static inline uint8_t get_4bit(void* ptr, int i) {
  uint8_t byte = ((uint8_t*)ptr)[i / 2];
  return (i % 2) ? (byte >> 4) : (byte & 0xf);
}

void BM168xEvaluator::staging_results(GlobalGenInterface& op) {
  for (auto v : op.getOperation()->getResults()) {
    auto addr = module::getAddress(v);
    auto name = module::getName(v).str();
    auto mem = mem_map[name];
    void* ptr = bm168x->get_system_mem_ptr(addr);
    if (!module::getStorageType(v).isInteger(4)) {
      std::memcpy(mem->data(), ptr, mem->size());
    } else {
      auto count = module::getNumElements(v);
      for (auto i = 0; i < count; ++i) {
        mem->at(i) = get_4bit(ptr, i);
      }
    }
  }
}

void BM168xEvaluator::staging_results(LocalGenInterface& op, local_sec_info_t sec_info) {
  assert(op.getOperation()->getNumResults() == 1);
  if (isa<tpu::LoadOp, tpu::StoreOp, tpu::YieldOp>(op))
    return;
  auto v = *(op.getOperation()->getResults().begin());
  const auto type = module::getStorageType(v);
  const int type_size = module::getDtypeSize(v);
  assert(!type.isInteger(4));
  auto ginfo = op.getGroupInfo((int64_t)0, (int64_t)0, (int64_t)0,
                               (int64_t)0, (int64_t)0);
  auto addr = ginfo.out_addr;
  int64_t N, C, D, H, W;
  module::getNCDHW(v, N, C, D, H, W, GROUP_NORMAL);
  auto name = module::getName(v).str();
  auto mem = mem_map[name];
  const int nidx = ginfo.n_idx;
  const int didx = ginfo.d_idx;
  const int cidx = sec_info.is_c_split ? sec_info.c_idx : 0;
  const int hidx = sec_info.is_h_split ? sec_info.out_h_idx : 0;
  const int widx = sec_info.is_w_split ? sec_info.out_w_idx : 0;
  const int nslice = sec_info.n_slice;
  const int dslice = sec_info.d_slice;
  const int cslice = sec_info.is_c_split ? sec_info.c_slice : C;
  const int hslice = sec_info.is_h_split ? sec_info.out_h_slice : H;
  const int wslice = sec_info.is_w_split ? sec_info.out_w_slice : W;
  int cstride = hslice * wslice;
  if (!type.isInteger(4)) {
    cstride = align_up(cstride, (BM168x::EU_BYTES / type_size));
  } else {
    cstride = align_up(cstride, BM168x::EU_BYTES * 2);
  }
  int nstride = cslice * cstride;
  int hstride = wslice;
  const int npu_num = BM168x::NPU_NUM;
  // save local data of shape (n, d, c, h, w) as shape (n, c, d, h, w)
  for (int n = 0; n < nslice; ++n) {
    for (int d = 0; d < dslice; ++d) {
      for (int c = 0; c < cslice; ++c) {
        if (wslice == W && !type.isInteger(4)) {
          const int offset = ((((n + nidx) * C + (c + cidx)) * D + (d + didx)) * H + hidx) * W;
          const int loc_offset = (n * dslice + d) * nstride + (c / npu_num) * cstride;
          void* ptr = bm168x->get_local_mem_ptr(c % npu_num, addr + loc_offset * type_size);
          assert((offset + hslice * wslice) * type_size <= mem->size());
          std::memcpy(mem->data() + offset * type_size, ptr, hslice * wslice * type_size);
        } else {
          for (int h = 0; h < hslice; ++h) {
            const int offset = ((((n + nidx) * C + (c + cidx)) * D + (d + didx)) * H + (h + hidx)) * W + widx;
            const int loc_offset = (n * dslice + d) * nstride + (c / npu_num) * cstride + h * hstride;
            if (!type.isInteger(4)) {
              void* ptr = bm168x->get_local_mem_ptr(c % npu_num, addr + loc_offset * type_size);
              assert((offset + wslice) * type_size <= mem->size());
              std::memcpy(mem->data() + offset * type_size, ptr, wslice * type_size);
            } else {
              void* ptr = bm168x->get_local_mem_ptr(c % npu_num, addr + loc_offset / 2);
              int8_t* data_ptr = mem->data() + offset;
              auto count = module::getNumElements(v);
              for (auto i = 0; i < count; ++i) {
                data_ptr[i] = get_4bit(ptr, i);
              }
            }
          }
        }
      }
    }
  }
}

void BM168xEvaluator::codegen(FuncOp funcOp) {

  funcOp.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (auto groupOp = dyn_cast<GroupOp>(op)) {
      Operation *prev_op = op->getPrevNode();
      while (prev_op && !isa<GroupOp, GlobalGenInterface>(prev_op)) {
        prev_op = prev_op->getPrevNode();
      }
      Operation *next_op = op->getNextNode();
      while (next_op && !isa<GroupOp, GlobalGenInterface>(next_op)) {
        next_op = next_op->getNextNode();
      }
      codegen_for_group(groupOp, prev_op, next_op);
      return WalkResult::skip();
    }

    if (auto globalOp = dyn_cast<GlobalGenInterface>(op)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "codegen op: '" << module::getName(globalOp) << "'\n");
      globalOp.codegen_global_bm168x();
      staging_results(globalOp);
    }
    return WalkResult::advance();
  });
}

void BM168xEvaluator::codegen_for_group(GroupOp gOp, Operation *prev_op,
                                        Operation *next_op) {

  auto nsecs = gOp.getNsecs();
  auto hsecs = gOp.getHsecs();
  auto dsecs = gOp.getDsecs();
  auto wsecs = gOp.getWsecs();
  auto csecs = gOp.getCsecs();
  auto swpipl_stage_num = gOp.getSwpiplStageNum();
  auto &body = gOp.getBody().front();
  auto flow = module::getI64Array(gOp.getFlow());
  auto group_type = static_cast<group_type_t>(gOp.getGroupType());
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
  // 2. create a vector to map id to op
  std::vector<Operation *> group_ops;
  for (int64_t id = 0; id < max_id;) {
    body.walk([&](Operation *op) {
      if (auto lgOp = dyn_cast<LocalGenInterface>(op)) {
        auto ginfo = lgOp.getGroupInfo((int64_t)0, (int64_t)0, (int64_t)0,
                                       (int64_t)0, (int64_t)0);
        if (ginfo.id == id) {
          group_ops.push_back(op);
          id++;
        }
      }
    });
  }
  // 3. recover overlap ops that will be executed in this group
  int64_t tmp_ts = 0;
  // <timestep_idx, prev_group_op>
  std::map<int64_t, std::vector<Operation *>> cur_other_downs;
  if (auto castOp = dyn_cast_or_null<GroupOp>(prev_op)) {
    auto other_down_overlap_op =
        module::getI64Array(gOp.getOtherDownOverlapOp());

    auto &prev_body = castOp.getBody().front();
    for (size_t i = 0; i < other_down_overlap_op->size(); ++i) {
      if (other_down_overlap_op->at(i) < 0) {
        tmp_ts = -other_down_overlap_op->at(i) - 1;
        cur_other_downs[tmp_ts] = std::vector<Operation *>();
      } else {
        int64_t id = other_down_overlap_op->at(i);
        prev_body.walk([&](Operation *op) {
          if (auto lgOp = dyn_cast<LocalGenInterface>(op)) {
            auto ginfo = lgOp.getGroupInfo((int64_t)0, (int64_t)0, (int64_t)0,
                                           (int64_t)0, (int64_t)0);
            if (ginfo.id == id) {
              cur_other_downs[tmp_ts].push_back(op);
            }
          }
        });
      }
    }
  }
  // <timestep_idx, next_group_op>
  std::map<int64_t, std::vector<Operation *>> cur_other_ups;
  if (auto castOp = dyn_cast_or_null<GroupOp>(next_op)) {
    auto other_up_overlap_op = module::getI64Array(gOp.getOtherUpOverlapOp());
    auto &next_body = castOp.getBody().front();
    for (size_t i = 0; i < other_up_overlap_op->size(); ++i) {
      if (other_up_overlap_op->at(i) < 0) {
        tmp_ts = -other_up_overlap_op->at(i) - 1;
        cur_other_ups[tmp_ts] = std::vector<Operation *>();
      } else {
        int64_t id = other_up_overlap_op->at(i);
        next_body.walk([&](Operation *op) {
          if (auto lgOp = dyn_cast<LocalGenInterface>(op)) {
            auto ginfo = lgOp.getGroupInfo((int64_t)0, (int64_t)0, (int64_t)0,
                                           (int64_t)0, (int64_t)0);
            if (ginfo.id == id) {
              cur_other_ups[tmp_ts].push_back(op);
            }
          }
        });
      }
    }
  }

  auto self_up_overlap_op = module::getI64Array(gOp.getSelfUpOverlapOp());
  auto self_down_overlap_op = module::getI64Array(gOp.getSelfDownOverlapOp());

  // 4. codegen for group
  int64_t stage_idx = 0;
  int64_t draining_idx = 0;
  bool draining_period = false;
  SoftwarePipeline timestep_swpipl;
  local_sec_info_t sec_info;
  int64_t timestep_num = timestep_table.size();

  for (uint64_t nstep = 0, cstep = 0, hstep = 0, dstep = 0, wstep = 0;
       nstep < nsecs || draining_period;) {
    /* add for software pipeline */
    timestep_swpipl.write_swloop_buffer(nstep, cstep, hstep, dstep, wstep,
                                        swpipl_stage_num);
    for (int64_t ts = 0; ts < timestep_num; ++ts) {
      bm168x->divide_sync_id();
      auto cur_op_ids = timestep_table[ts];
      for (auto id : cur_op_ids) {

        auto lgOp = cast<LocalGenInterface>(group_ops[id]);
        auto ginfo = lgOp.getGroupInfo(nstep, hstep, dstep, wstep, cstep);
        if ((!draining_period && ginfo.stage > stage_idx) ||
            (draining_period &&
             (ginfo.stage < draining_idx || ginfo.stage > stage_idx))) {
          continue;
        }
        const tensor_step_t *tensor_step =
            timestep_swpipl.read_swloop_buffer(ginfo.stage);

        // only consider first loop load
        if (stage_idx == 0 &&
            std::find(self_up_overlap_op->begin(), self_up_overlap_op->end(),
                      id) != self_up_overlap_op->end()) {
          continue;
        }
        // only consider last loop store
        if (draining_period && draining_idx == 2 &&
            std::find(self_down_overlap_op->begin(),
                      self_down_overlap_op->end(),
                      id) != self_down_overlap_op->end()) {
          continue;
        }

        ginfo = lgOp.getGroupInfo(tensor_step->nstep, tensor_step->hstep,
                                  tensor_step->dstep, tensor_step->wstep,
                                  tensor_step->cstep);
        if (ginfo.overstepped == false || stage_idx == ginfo.stage) {
          ginfo.overstepped = true;
          lgOp.assign_sec_info(tensor_step->nstep, tensor_step->cstep,
                               tensor_step->hstep, tensor_step->dstep,
                               tensor_step->wstep, group_type, sec_info);
          LLVM_DEBUG(llvm::dbgs()
                     << "codegen op: '" << module::getName(lgOp) << "'\n");
          lgOp.codegen_local_bm168x(tensor_step->nstep, tensor_step->cstep,
                                    tensor_step->hstep, tensor_step->dstep,
                                    tensor_step->wstep, group_type, sec_info);
          staging_results(lgOp, sec_info);
        }
      } // ops, include Load/Store op

      // process overlap ops
      bool first_compute_loop = stage_idx == 1;
      bool last_compute_loop = (draining_period && draining_idx == 1);
      codegen_for_overlap_ops(cur_other_downs, cur_other_ups, prev_op, next_op,
                              ts, first_compute_loop, last_compute_loop);

      bm168x->merge_sync_id();
    } // timestep

    if (!draining_period) {
      cstep++;
      if (cstep >= csecs) {
        cstep = 0;
        wstep++;
      }
      if (wstep >= wsecs) {
        wstep = 0;
        hstep++;
      }
      if (hstep >= hsecs) {
        hstep = 0;
        dstep++;
      }
      if (dstep >= dsecs) {
        dstep = 0;
        nstep++;
        if (nstep >= nsecs) { // && swpipl_stage_num > 1
          draining_period = true;
        }
      }
    }
    stage_idx++;
    if (draining_period) {
      draining_idx++;
      if (draining_idx >= swpipl_stage_num) {
        draining_period = false;
        stage_idx = 0;
        draining_idx = 0;
      }
    }
  }
}

void BM168xEvaluator::codegen_for_overlap_ops(
    std::map<int64_t, std::vector<Operation *>> cur_other_downs,
    std::map<int64_t, std::vector<Operation *>> cur_other_ups,
    Operation *prev_op, Operation *next_op, int64_t cur_ts,
    bool first_compute_loop, bool last_compute_loop) {

  local_sec_info_t sec_info;
  if (last_compute_loop) {
    auto iter = cur_other_ups.find(cur_ts);
    if (iter != cur_other_ups.end()) {
      auto castOp = cast<GroupOp>(next_op);
      auto next_group_type = static_cast<group_type_t>(castOp.getGroupType());
      auto &cur_ops = iter->second;
      for (auto op : cur_ops) {
        auto lgOp = cast<LocalGenInterface>(op);
        auto pid_node = (CMD_ID_NODE *)(*BM168x::instance())->bdc_node;
        if (isa<LoadOp, StoreOp>(op)) {
          pid_node = (CMD_ID_NODE *)(*BM168x::instance())->gdma_node;
        }
        lgOp.assign_sec_info(0l, 0l, 0l, 0l, 0l, next_group_type, sec_info);
        LLVM_DEBUG(llvm::dbgs()
                   << "codegen op: '" << module::getName(lgOp) << "'\n");
        lgOp.codegen_local_bm168x(0l, 0l, 0l, 0l, 0l, next_group_type,
                                  sec_info);
        staging_results(lgOp, sec_info);
      }
    }
  }

  if (first_compute_loop) {
    auto iter = cur_other_downs.find(cur_ts);
    if (iter != cur_other_downs.end()) {
      auto castOp = cast<GroupOp>(prev_op);
      auto prev_group_type = static_cast<group_type_t>(castOp.getGroupType());
      auto nsecs = castOp.getNsecs();
      auto hsecs = castOp.getHsecs();
      auto dsecs = castOp.getDsecs();
      auto wsecs = castOp.getWsecs();
      auto csecs = castOp.getCsecs();
      auto &cur_ops = iter->second;
      for (auto op : cur_ops) {
        auto lgOp = cast<LocalGenInterface>(op);
        lgOp.assign_sec_info(nsecs - 1, csecs - 1, hsecs - 1, dsecs - 1,
                             wsecs - 1, prev_group_type, sec_info);
        LLVM_DEBUG(llvm::dbgs()
                   << "codegen op: '" << module::getName(lgOp) << "'\n");
        lgOp.codegen_local_bm168x(nsecs - 1, csecs - 1, hsecs - 1, dsecs - 1,
                                  wsecs - 1, prev_group_type, sec_info);
        staging_results(lgOp, sec_info);
      }
    }
  }
}

void BM168xEvaluator::CreateSubNet(func::CallOp call) {
  auto func = module::getFuncOp(module, call.getCallee());
  codegen(func);
}

} // namespace tpu
} // namespace tpu_mlir
