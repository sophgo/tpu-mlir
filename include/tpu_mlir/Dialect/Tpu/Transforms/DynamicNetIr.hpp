//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#pragma once
#include <list>
#include <map>
#include <set>
#include <vector>
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Backend/BM168x/BM168x.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/IrInfo.hpp"
#include "tpu_mlir/Dialect/Tpu/Transforms/DynamicLayer.hpp"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LayerGroupUtil.h"
using namespace std;
using namespace llvm;
namespace tpu_mlir {
namespace tpu {

typedef struct stage_param{
  u32 ir_info_len;
  int height_high;
  int height_low;
  int width_high;
  int width_low;
}stage_param_t;

class SubnetIr {
  public:
    explicit SubnetIr(StringRef chip_name, int version) : chip(chip_name), dynamic_version(version) {fw_ir_length = 0;}
    ~SubnetIr () {clear_all();}
    void generate_compiler_ir(ModuleOp &module, func::CallOp &call,
                              std::function<void(Operation *, SubnetIr*)> task);

    int write_binary_ir_to_buffer();
    u32 get_fw_ir_length() {return fw_ir_length;}

    void clear_all();
    int get_dynamic_version();
    friend class DynCodegenPass;
  protected:
    //=========================
    //functions
    //=========================
    void generate_group_time_step_ir(Operation *op);

    void generate_crop_layer_shape_tensor_record();
    void global_layer_ir_generate(Operation *op);
    void global_layer_ir_generate_v2(Operation *op);

    void local_layer_ir_generate();
    void local_layer_ir_generate_v2(Operation *op);

    void gdma_tensor_ir_generate(Operation *op,
                                  vector<ir_tensor_gdma_info_t>& tensor_gdma_info_v1,
                                  group_type_t group_type,
                                  bool swpipl_enable,
                                  int stage, uint64_t local_addr);

    int write_ir_to_buffer(void* ir_buffer,
                           vector<unsigned int>& input_tensor_ids,
                           vector<unsigned int>& output_tensor_ids,
                           int group_start, int group_end);

    void* write_local_layer_info_buffer(void* p_ir_buf, ir_layer_info_t* p_ir_layer_info);
    void* write_local_layer_info_buffer_v2(void* p_ir_buf, Operation *op, FW_LAYER_TYPE_T fw_type, shared_ptr<BasicTimeStep> time_step);
    void* write_tensor_gdma_info_buffer(void* p_ir_buf, ir_tensor_gdma_info_t* ir_tensor_gdma_info);
    void* write_global_layer_info_buffer(void* p_ir_buf, ir_layer_info_t* ir_layer_info);
    void* write_global_layer_info_buffer_v2(void* p_ir_buf, Operation *op, FW_LAYER_TYPE_T fw_type);

    bool load_from_weightop(Value& v);
    bool loadOp_and_load_from_weightop(Operation *op);
    uint32_t get_tensor_group_consumer_num(Value v);
    void get_fw_input_tensor_info(
                                  LgInfo & group_ops,
                                  int hsecs,
                                  map<int, dynamic_tensor_info_t>& tensor_to_dynamic_info);

    bool strip_back_judge(Value v, const LgInfo &lg_info,
                      const std::multiset<Operation *> &op_set,
                      const std::set<Value, value_cmp> &out_tensor_set);
    void layer_data_back_dynamic_info(
                    LgInfo & group_ops,
                    Value tensor,
                    list<Value>& tensor_branchs,
                    map<int, dynamic_tensor_info_t>& tensor_to_dynamic_info,
                    std::multiset<Operation *>& layer_set, const set<Value, value_cmp>& out_tensor_set);

    void get_neuron_timestep_consumer(map<int, int>& tensor_to_consumer_num, shared_ptr<BasicTimeStep> time_step);
    void insert_produced_tensors(map<int, int>& tensor_to_consumer_num,
                                    int tensor_id) ;
    //=========================
    //IR information
    //=========================
    uint32_t fw_ir_length;  //unit: byte
    //net related ir information
    vector<uint32_t> net_input_tensor_id;
    vector<uint32_t> net_output_tensor_id;

    //The following information relates to layer group
    vector<fw_timestep_base_info_t>                 ir_group_timestep_base_info;                    //size is equal to layer group num
    vector<vector<fw_input_tensor_info_t> >         ir_group_input_tensor_info;                     //1st dim: group num, 2nd dim: input num
    vector<vector<uint32_t> >                            ir_group_out_tensor_id_and_consumer_num;        //1st dim: group num, 2nd dim: output num

    //group time step related ir information
    vector<vector<vector<ir_layer_info_t> > >       ir_group_timestep_layer_param;                  //1st dim: group num, 2nd dim: timestep number, 3rd dim: layer_num
    vector<vector<vector<ir_tensor_gdma_info_t> > > ir_group_timestep_tensor_gdma_param;            //1st dim: group num, 2nd dim: timestep number, 3rd dim: tensor_gdma_num
    vector<vector<uint32_t> >                             ir_group_extra_tensor_record;

    vector<uint32_t>                                            m_ir_buffer;
    vector<stage_param_t>                                       stage_param_vv;
    // subnet related param
    vector<LgInfo> m_layer_groups_;
    vector<std::shared_ptr<BasicTimeStep>> m_time_step_groups_;
    map<Operation *, dynamic_layer*> dynamic_layers_;
    StringRef chip;
    int dynamic_version;
    func::CallOp *subnet_;
};

}
}
