//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"

#include "tpu_mlir/Support/Module.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>

using namespace tpu_mlir::backend;

void tpu::ActiveOp::codegen_global_bm1684() {
    // active_global_spec_t
    uint64_t bottom_global_offset = module::getAddress(getInput());
    uint64_t top_global_offset = module::getAddress(getResult());
    int64_t n, c, h, w;
    module::getNCHW(getInput(), n, c, h, w);
    uint64_t length = n*c*h*w;

    float coeffs[8];
    if (getCoeffs().has_value()){
        const auto coeffs_ = module::getF64Array(getCoeffs().value());
        for (int i = 0; i < coeffs_->size(); ++i){
            coeffs[i] = (float)coeffs_->at(i);
        }
    }
    
    int activate_type = (int)getMode();
    switch(getMode()){
       case ActiveMode::EXP:
       case ActiveMode::SIGMOID: break;
       case ActiveMode::SILU:
           activate_type = (int)ActiveMode::SWISH;
           coeffs[0] = 1.0;
           break;
       default:
           llvm_unreachable("Not Implement such activate type, please add it.");
           break;
    } 
    if (module::isUniformQuantized(getOutput())){
        llvm_unreachable("Not Implemented!");
    }else{
        BM1684::instance().dl_nodechip_active_forward_parallel(
                bottom_global_offset, top_global_offset, length,
                activate_type, coeffs, (CMD_ID_NODE *)BM1684::instance().cmdid_node);
    }

}

int64_t tpu::ActiveOp::getBufferSize_bm1684(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {

    int64_t buffer_size = 0;
    int64_t tensor_size = in_lmem_bytes;
    if (!module::isUniformQuantized(getOutput())){
        switch(getMode()){
            case ActiveMode::EXP:
            case ActiveMode::SIGMOID: buffer_size = tensor_size; break;
            case ActiveMode::SILU:
                buffer_size = 2 * tensor_size;
                break;
            default:
                llvm_unreachable("Not Implement such activate type, please add it.");
                break;
        }
    }

  return buffer_size;
}

void tpu::ActiveOp::codegen_local_bm1684(int64_t n_step, int64_t h_step, local_sec_info_t &sec_info) {
    auto out_g_info = getGroupInfo(n_step, h_step);
    auto in_g_info = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);
    
    int64_t n=1, c=1, h=1, w=1, depth=1;
    uint32_t bottom_dim[MAX_SHAPE_DIMS] = {1};
    if (out_g_info.type == GROUP_NORMAL){
        module::getNCHW(getInput(), n, c, h, w);
        bottom_dim[0] = (uint32_t)in_g_info.n_slice;
        bottom_dim[1] = (uint32_t)c;
        bottom_dim[2] = (uint32_t)in_g_info.h_slice;
        bottom_dim[3] = (uint32_t)w;
    }else if (out_g_info.type == GROUP_3D){   
        module::getNCDHW(getInput(), n, c, depth, h, w, GROUP_3D);
        bottom_dim[0] = (uint32_t)in_g_info.n_slice;
        bottom_dim[1] = (uint32_t)c;
        bottom_dim[2] = (uint32_t)in_g_info.h_slice;
        bottom_dim[3] = (uint32_t)w;
    }else{
        llvm_unreachable("BM1684 do not support such LayerGroup method");
    }
    uint32_t bottom_local_offset = in_g_info.out_addr;
    uint32_t top_local_offset = out_g_info.out_addr;
    uint32_t imm_buffer_local_offet = out_g_info.buffer_addr;
 
    int activate_type = (int)getMode();
    float prelu_slope = 0.0;
    switch(getMode()){
        case ActiveMode::EXP:
        case ActiveMode::SIGMOID: break;
        case ActiveMode::SILU: activate_type = (int)ActiveMode::SWISH; prelu_slope = 1.0; break;
        default:
            llvm_unreachable("Not Implement such activate type");
            break;
    }
    if (module::isUniformQuantized(getOutput())){
        llvm_unreachable("NOT SUPPORT local fix8b layer");
    }else{
        BM1684::instance().dl_nodechip_active_forward_local(
                bottom_local_offset, top_local_offset, imm_buffer_local_offet,
                bottom_dim[0] * depth, bottom_dim[1], bottom_dim[2], bottom_dim[3],
                activate_type, &prelu_slope, (CMD_ID_NODE *)BM1684::instance().bdc_node
                );
    }
}
