//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <vector>
#include "string.h"
#include <map>
#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/BM1684DynIrUtils.hpp"
#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynCompileCommon.hpp"
#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynIrInfo.hpp"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LayerGroupUtil.h"

using namespace std;
namespace tpu_mlir {
namespace tpu {
struct value_cmp {
  bool operator()(Value v0, Value v1) const {
    if (module::getName(v0).str() < module::getName(v1).str()) {
      return true;
    }
    return false;
  }
};

extern void DynCodegenInit();
extern int get_tensor_id(Value v);
extern bool is_net_input(Value v);
extern bool is_net_output(Value v);
uint32_t push_back_layer_global_tensor(Value v, vector<ir_tensor_info_t>& ir_tensor_info_v, bool is_layer_in);
void dynamic_push_back_local_tensor(vector<ir_tensor_info_t> &ir_tensor_info_v, Value v);
void dynamic_push_back_local_buffer(vector<ir_tensor_info_t> &ir_tensor_info_v, int tensor_id, Value output);
void dynamic_common_ir_layer_info(ir_layer_info_t* ir_layer_info, Value input, Value output);
enum DynamicTensorType
{
    DYNAMIC_NEURON = 0,
    DYNAMIC_COEFF = 1,
    DYNAMIC_SHAPE = 2
};
struct dynamic_global_tensor_spec {
    uint8_t type;
    uint32_t id;
    uint32_t dtype;
    uint64_t addr;
    uint8_t dims;
    int32_t shape[MAX_SHAPE_DIMS];
    uint8_t is_net_io;
    int*    host_data;
    int     elem_num;
};

struct dynamic_local_tensor_spec {
    uint8_t type;
    uint32_t id;
    uint32_t dtype;
    uint32_t addr;
    uint8_t dims;
    int32_t shape[MAX_SHAPE_DIMS];
    uint8_t consume_num;
    int*    host_data;
    int     elem_num;
};

class dynamic_layer {
public:
    enum IOType {
        INPUT = 0,
        OUTPUT = 1
    };

    explicit dynamic_layer(Operation *op):op_(op) {}
    //for BM1684X
    size_t get_global_ir_length();
    size_t get_local_ir_length();
    //for BM1684
    uint32_t get_global_ir_length(ir_layer_info_t *ir_layer_info);
    int32_t get_local_ir_length(ir_layer_info_t *ir_layer_info);
    int global_ir_version()
    {
        return 0;
    }

    int local_ir_version()
    {
        return 0;
    }

    size_t write_local_ir(
        void *buffer,
        const std::map<int, int> &consume_table);
    size_t write_global_ir(void *buffer);
protected:
    size_t write_global_ir_impl(void *buffer, bool feign = false);
    size_t write_local_ir_impl(
        void *buffer,
        const std::map<int, int> &consume_table,
        bool feign = false);

    std::vector<dynamic_global_tensor_spec> get_input_global_tensor_specs();
    std::vector<dynamic_global_tensor_spec> get_output_global_tensor_specs();
    std::vector<dynamic_local_tensor_spec> get_input_local_tensor_specs();
    std::vector<dynamic_local_tensor_spec> get_output_local_tensor_specs();
    size_t write_global_tensor_specs(void *buffer, bool feign);
    size_t write_local_tensor_specs(
        void *buffer,
        const std::map<int, int> &consume_table,
        bool feign);

    template<typename T>
    size_t copy_to_buffer(void *buffer, const T &spec)
    {
        auto p = static_cast<char *>(buffer);
        memcpy(p, &spec, sizeof(spec));
        p += sizeof(spec);
        return p - static_cast<char *>(buffer);
    }

    template<IOType io, typename T>
    size_t copy_tensors_to_buffer(
        void *buffer,
        const T *,
        size_t n,
        bool feign = false);
private:
    Operation *op_;
};

}
}
