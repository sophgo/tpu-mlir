#pragma once

#include <cstdint>
#include <cstddef>

extern "C"
{
    enum tpudnnStatus_t
    {
        TPUDNN_STATUS_SUCCESS,
        TPUDNN_STATUS_FAILED
    };

    typedef void *tpudnnHandle_t;

    tpudnnHandle_t tpudnnCreate(int deviceID = 0);
    void tpudnnDestroy(tpudnnHandle_t handle);

    tpudnnStatus_t tpudnnFlush(tpudnnHandle_t handle);

    void *tpudnnPhysToVirt(tpudnnHandle_t handle, uint64_t addr);
    uint64_t tpudnnVirtToPhys(tpudnnHandle_t handle, void *addr);

    tpudnnStatus_t tpudnnSetupC2CTopology();
    tpudnnStatus_t tpudnnGetC2CTopology(int *chipNum, const int **outTopology);
    tpudnnStatus_t tpudnnCheckChipMap(int world_size, int *chipMap);
    tpudnnStatus_t tpudnnGetC2CRing(int world_size, int *chipMap);
    tpudnnStatus_t tpudnnGetUniqueId(tpudnnHandle_t handle, char* uniqueId);
    tpudnnHandle_t tpudnnHandleFromStream(int deviceID, void* stream, void* module);
    tpudnnStatus_t tpudnnLaunchKernel(tpudnnHandle_t handle, const char *kernelName,
                                      void *arg, size_t argBytes, int groupNum, int groupSize);
    tpudnnStatus_t tpudnnSync(tpudnnHandle_t handle);

    tpudnnStatus_t tpudnnMemcpyHostToDevice(tpudnnHandle_t handle, uint64_t dst, void *src, size_t size);
    tpudnnStatus_t tpudnnMemcpyDeviceToHost(tpudnnHandle_t handle, void *dst, uint64_t src, size_t size);

    typedef void *tpudnnGraph_t;
    tpudnnStatus_t tpudnnStreamBeginCapture(tpudnnHandle_t handle);
    tpudnnStatus_t tpudnnStreamEndCapture(tpudnnHandle_t handle, tpudnnGraph_t *graph);
    tpudnnStatus_t tpudnnGraphDestroy(tpudnnGraph_t graph);
    typedef void *tpudnnGraphExec_t;
    tpudnnStatus_t tpudnnGraphInstantiate(tpudnnGraphExec_t *instance, tpudnnGraph_t graph, void *ext);
    tpudnnStatus_t tpudnnGraphExecDestroy(tpudnnGraphExec_t instance);
    tpudnnStatus_t tpudnnGraphLaunch(tpudnnGraphExec_t instance, tpudnnHandle_t handle);
}

#include "tpuDNNTensor.h"
