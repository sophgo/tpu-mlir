#pragma once

#include <unistd.h>
#include <stdint.h>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <cassert>
#ifdef __cplusplus
extern "C" {
#endif

/* Wrapper of runtime APIs */

/**
 * @brief TPU runtime error codes enumeration
 */
typedef enum {
  tpuSuccess = 0,                 ///< Operation completed successfully
  tpuErrorInvalidDevice,          ///< Invalid device identifier
  tpuErrorNoDevice,               ///< No device found
  tpuErrorFailure,                ///< General operation failure
  tpuErrorTimeout,                ///< Operation timed out
  tpuErrorInvalidParam,           ///< Invalid parameter provided
  tpuErrorOutOfMemory,            ///< Insufficient memory available
  tpuErrorNotSupported,           ///< Operation not supported
  tpuErrorInvalidValue,           ///< Invalid value provided
  tpuErrorMemoryAllocation,       ///< Memory allocation error
  tpuErrorKernelModuleLoad        ///< Kernel module loading error
} tpuError_t;

/**
 * @brief TPU stream handle type
 */
typedef void* tpuStream_t;

/**
 * @brief TPU event handle type
 */
typedef void* tpuEvent_t;

/**
 * @brief TPU kernel module handle type
 */
typedef void* tpuKernelModule_t;

/**
 * @brief TPU topology structure information
 */
typedef struct {
  uint32_t device_num;    ///< Number of devices
  uint16_t parent_dev;   ///< Parent device ID
  uint16_t child_dev;     ///< Child device ID
  uint8_t parent_pcie;   ///< Parent device PCIe information
  uint8_t child_pcie;     ///< Child device PCIe information
  int8_t parent_port;    ///< Parent device port
  int8_t child_port;      ///< Child device port
} tpuTopology_t;

/**
 * @brief Initialize TPU runtime environment
 * @return tpuError_t Error code
 */
tpuError_t tpuInit();

/**
 * @brief Get the number of available TPU devices
 * @param count[out] Pointer to store device count
 * @return tpuError_t Error code
 */
tpuError_t tpuGetDeviceCount(int* count);

/**
 * @brief Get current device ID
 * @param device[out] Pointer to store device ID
 * @return tpuError_t Error code
 */
tpuError_t tpuGetDevice(int* device);

/**
 * @brief Set the current TPU device to use
 * @param device Device ID
 * @return tpuError_t Error code
 */
tpuError_t tpuSetDevice(int device);

/**
 * @brief Create a TPU stream
 * @param stream[out] Pointer to store stream handle
 * @return tpuError_t Error code
 */
tpuError_t tpuStreamCreate(tpuStream_t* stream);

/**
 * @brief Destroy a TPU stream
 * @param stream Stream handle
 * @return tpuError_t Error code
 */
tpuError_t tpuStreamDestroy(tpuStream_t stream);

/**
 * @brief Synchronize and wait for all operations in the stream to complete
 * @param stream Stream handle
 * @return tpuError_t Error code
 */
tpuError_t tpuStreamSynchronize(tpuStream_t stream);

/**
 * @brief Make stream wait for a specific event to complete
 * @param stream Stream handle
 * @param event Event handle
 * @return tpuError_t Error code
 */
tpuError_t tpuStreamWaitEvent(tpuStream_t stream, tpuEvent_t event);

/**
 * @brief Create a TPU event
 * @param event[out] Pointer to store event handle
 * @return tpuError_t Error code
 */
tpuError_t tpuEventCreate(tpuEvent_t* event);

/**
 * @brief Free a TPU event
 * @param event Event handle
 * @param stream Stream handle
 * @return tpuError_t Error code
 */
tpuError_t tpuEventDestroy(tpuEvent_t event, tpuStream_t stream);

/**
 * @brief Record an event on a specific stream
 * @param event Event handle
 * @param stream Stream handle
 * @return tpuError_t Error code
 */
tpuError_t tpuEventRecord(tpuEvent_t event, tpuStream_t stream);

/**
 * @brief Query event status
 * @param event Event handle
 * @return tpuError_t Error code
 */
tpuError_t tpuEventQuery(tpuEvent_t event);

/**
 * @brief Calculate elapsed time between two events
 * @param ms[out] Elapsed time in milliseconds
 * @param start Start event
 * @param end End event
 * @return tpuError_t Error code
 */
tpuError_t tpuEventElapsedTime(float* ms, tpuEvent_t start, tpuEvent_t end);

/**
 * @brief Synchronize and wait for event to complete
 * @param event Event handle
 * @return tpuError_t Error code
 */
tpuError_t tpuEventSynchronize(tpuEvent_t event);

/**
 * @brief Load TPU kernel module from memory data
 * @param data Pointer to kernel module data
 * @param size Data size in bytes
 * @param stream Stream handle
 * @return tpuKernelModule_t Kernel module handle
 */
tpuKernelModule_t tpuKernelModuleLoad(const char* data, size_t size, tpuStream_t stream);

/**
 * @brief Load TPU kernel module from file
 * @param file File path
 * @param stream Stream handle
 * @return tpuKernelModule_t Kernel module handle
 */
tpuKernelModule_t tpuKernelModuleLoadFromFile(const char* file, tpuStream_t stream);

/**
 * @brief Unload TPU kernel module
 * @param module Kernel module handle
 * @param stream Stream handle
 * @return tpuError_t Error code
 */
tpuError_t tpuKernelUnloadModule(tpuKernelModule_t module, tpuStream_t stream);

/**
 * @brief Unload TPU kernel module
 * @param module Kernel module handle
 * @param stream Stream handle
 * @return tpuError_t Error code
 */
tpuError_t tpuKernelUnloadModule(tpuKernelModule_t module, tpuStream_t stream);

/**
 * @brief Launch TPU kernel synchronously
 * @param module Kernel module handle
 * @param kernel_name Kernel name
 * @param args Kernel arguments
 * @param arg_size Arguments size
 * @param group_num Group number
 * @param group_size Group size, like thread-num of cuda with each group
 * @param stream Stream handle
 * @return tpuError_t Error code
 */
tpuError_t tpuKernelLaunch(tpuKernelModule_t module, const char* kernel_name,
                           const void* args, uint32_t arg_size,
                           uint64_t group_num, uint64_t group_size,
                           tpuStream_t stream);

/**
 * @brief Launch TPU kernel asynchronously
 * @param module Kernel module handle
 * @param kernel_name Kernel name
 * @param args Kernel arguments
 * @param arg_size Arguments size
 * @param group_num Group number
 * @param group_size Group size, like thread-num of cuda with each group
 * @param stream Stream handle
 * @return tpuError_t Error code
 */
tpuError_t tpuKernelLaunchAsync(tpuKernelModule_t module,
                                const char* kernel_name, const void* args,
                                uint32_t arg_size, uint64_t group_num,
                                uint64_t group_size, tpuStream_t stream);

/**
 * @brief Allocate memory on device
 * @param devPtr[out] Pointer to store device memory address
 * @param size Memory size in bytes
 * @return tpuError_t Error code
 */
tpuError_t tpuMalloc(void** devPtr, size_t size);

/**
 * @brief Free memory on device
 * @param devPtr Device memory pointer
 * @return tpuError_t Error code
 */
tpuError_t tpuFree(void* devPtr);

/**
 * @brief Asynchronous device to device memory copy
 * @param dstPtr Destination device pointer
 * @param srcPtr Source device pointer
 * @param size Copy size in bytes
 * @param stream Stream handle
 * @return tpuError_t Error code
 */
tpuError_t tpuMemcpyD2DAsync(void* dstPtr, const void* srcPtr, size_t size, tpuStream_t stream);

/**
 * @brief Asynchronous host to device memory copy
 * @param devPtr Device pointer
 * @param hostPtr Host pointer
 * @param size Copy size in bytes
 * @param stream Stream handle
 * @return tpuError_t Error code
 */
tpuError_t tpuMemcpyH2DAsync(void* devPtr, const void* hostPtr, size_t size, tpuStream_t stream);

/**
 * @brief Asynchronous device to host memory copy
 * @param hostPtr Host pointer
 * @param devPtr Device pointer
 * @param size Copy size in bytes
 * @param stream Stream handle
 * @return tpuError_t Error code
 */
tpuError_t tpuMemcpyD2HAsync(void* hostPtr, const void* devPtr, size_t size, tpuStream_t stream);

/**
 * @brief Synchronous device to device memory copy
 * @param dstPtr Destination device pointer
 * @param srcPtr Source device pointer
 * @param size Copy size in bytes
 * @return tpuError_t Error code
 */
tpuError_t tpuMemcpyD2D(void* dstPtr, const void* srcPtr, size_t size);

/**
 * @brief Synchronous host to device memory copy
 * @param devPtr Device pointer
 * @param hostPtr Host pointer
 * @param size Copy size in bytes
 * @return tpuError_t Error code
 */
tpuError_t tpuMemcpyH2D(void* devPtr, const void* hostPtr, size_t size);

/**
 * @brief Synchronous device to host memory copy
 * @param hostPtr Host pointer
 * @param devPtr Device pointer
 * @param size Copy size in bytes
 * @return tpuError_t Error code
 */
tpuError_t tpuMemcpyD2H(void* hostPtr, const void* devPtr, size_t size);

/**
 * @brief Setup TPU topology configuration
 * @return tpuError_t Error code
 */
tpuError_t tpuSetupTopology();

/**
 * @brief Get TPU topology information
 * @param topology[out] Pointer to store topology information
 * @return tpuError_t Error code
 */
tpuError_t tpuGetTopology(tpuTopology_t** topology);

/**
 * @brief Get total memory size of the device
 * @param allMemSize[out] Pointer to store total memory size in bytes
 * @return tpuError_t Error code
 */
tpuError_t tpuGetAllMemory(size_t* allMemSize);

/**
 * @brief Get free memory size of the device
 * @param freeMemSize[out] Pointer to store free memory size in bytes
 * @return tpuError_t Error code
 */
tpuError_t tpuGetFreeMemory(size_t* freeMemSize);

/**
 * @brief Get peak memory usage size of the device
 * @param peakMemSize[out] Pointer to store peak memory size in bytes
 * @return tpuError_t Error code
 */
tpuError_t tpuGetPeakMemory(size_t* peakMemSize);

/**
 * @brief Reset peak memory usage size of the device
 * @return tpuError_t Error code
 */
tpuError_t tpuResetPeakMemory();


typedef void *tpuGraph_t;
typedef void *tpuGraphExec_t;
tpuError_t tpuStreamBeginCapture(tpuStream_t capture_stream_);
tpuError_t tpuStreamEndCapture(tpuStream_t capture_stream_, tpuGraph_t* graph_);
tpuError_t tpuGraphDestroy(tpuGraph_t graph_);
tpuError_t tpuGraphExecDestroy(tpuGraphExec_t graph_exec_);
tpuError_t tpuGraphInstantiate(tpuGraphExec_t* graph_exec_, tpuGraph_t graph_, void *ext);
tpuError_t tpuGraphLaunch(tpuGraphExec_t graph_exec_, tpuStream_t capture_stream_);

#ifdef __cplusplus
}
#endif
