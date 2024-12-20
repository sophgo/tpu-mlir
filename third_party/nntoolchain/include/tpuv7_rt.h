/* SPDX-License-Identifier: GPL-2.0 */
#ifndef __TPUV7_H__
#define __TPUV7_H__

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MD5SUM_LEN 16
#define LIB_MAX_NAME_LEN 64
#define FUNC_MAX_NAME_LEN 64

typedef enum {
	tpuRtSuccess = 0,
	tpuRtDevnotready = 1, /* Device not ready yet */
	tpuRtErrFailure = 2,     /* General failure */
	tpuRtErrTimeout = 3,     /* Timeout */
	tpuRtErrParam = 4,       /* Parameters invalid */
	tpuRtErrNomem = 5,       /* Not enough memory */
	tpuRtErrData = 6,        /* Data error */
	tpuRtErrBusy = 7,        /* Busy */
	tpuRtErrNoFeature = 8,   /* Not supported yet */
	tpuRtErrNotSupported = 9,
	tpuRtErrorInitializationError,
	tpuRtErrorNoDevice,
	tpuRtErrorNoStream,
	tpuRtErrorInvalidValue,
	tpuRtErrorInvalidResourceHandle,
	tpuRtErrorMemoryAllocation
} tpuRtStatus_t;

typedef struct tpuRtModule {
	// void *lib_handle;
	char lib_name[LIB_MAX_NAME_LEN];
	unsigned char md5[MD5SUM_LEN];
} tpuRtModule;

struct tpuRtCheckModule {
	unsigned char md5[MD5SUM_LEN];
	int loaded;
};

typedef struct tpuRtLaunchOutput {
	void *output_args;
	uint32_t output_size;
	int ret;
} *tpuRtLaunchOutput_t;

typedef struct tpuRtModule *tpuRtKernelModule_t;
typedef struct tpuRtStream *tpuRtStream_t;
typedef struct tpuRtEvent *tpuRtEvent_t;
typedef struct tpuRtLaunchOutput *tpuRtLaunchOutput_t;
typedef struct timeval tpuRtTimeRecord;
typedef int (*pTpuRtStreamCallback)(void *);

struct c2c_port_info {
	uint32_t chip_num;
	uint16_t src_device_id;
	uint16_t dst_device_id;
	uint8_t src_pcie_id;
	uint8_t dst_pcie_id;
	int8_t send_port;
	int8_t recv_port;
};

void sgTimeRecord(tpuRtTimeRecord *time);
uint32_t sgTimeCalculate(tpuRtTimeRecord time_start, tpuRtTimeRecord time_end);

/**
 * @name    tpuRtInit
 * @brief	To init runtime
 * @ingroup tpuv7_rt
 *
 * @retval  tpuRtSuccess  Succeeds.
 *          Other code Fails.
 */
tpuRtStatus_t tpuRtInit(void);

/**
 * @name    tpuRtGetDeviceCount
 * @brief   To get device number
 * @ingroup tpuv7_rt
 *
 * @param [out]	count  device number
 * @retval	puRtSuccess  Succeeds.
 *          Other code Fails.
 */
tpuRtStatus_t tpuRtGetDeviceCount(int *count);

/**
 * @name    tpuRtGetDevice
 * @brief   To get device id in current pid/tid
 * @ingroup tpuv7_rt
 *
 * @param [out]  device	device id
 * @retval  tpuRtSuccess  Succeeds.
 *          Other code Fails.
 */
tpuRtStatus_t tpuRtGetDevice(int *device);

/**
 * @name    tpuRtSetDevice
 * @brief   To set device id in current pid/tid
 * @ingroup tpuv7_rt
 *
 * @param [in]  device	device id
 * @retval  tpuRtSuccess  Succeeds.
 *          Other code Fails.
 */
tpuRtStatus_t tpuRtSetDevice(int device);

/**
 * @name    tpuRtDeviceSynchronize
 * @brief   To sync current device
 * @ingroup tpuv7_rt
 *
 * @retval  tpuRtSuccess  Succeeds.
 *          Other code Fails.
 */
tpuRtStatus_t tpuRtDeviceSynchronize(void);

/**
 * @name    tpuRtMalloc
 * @brief   To malloc device memory
 * @ingroup tpuv7_rt
 *
 * @param [out]	devPtr	memory array
 * @param [in]  size	memory size
 * @param [in]  parallel_num	parallelism
 * @retval  tpuRtSuccess  Succeeds.
 *          Other code Fails.
 */
tpuRtStatus_t tpuRtMalloc(void **devPtr, unsigned long long size, int parallel_num);

/**
 * @name    tpuRtMallocMedia
 * @brief   To malloc device memory for media
 * @ingroup tpuv7_rt
 *
 * @param [out] devPtr	memory array
 * @param [in]  size	memory size
 * @retval  tpuRtSuccess  Succeeds.
 *          Other code Fails.
 */
tpuRtStatus_t tpuRtMallocMedia(void **devPtr, unsigned long long size);

/**
 * @name    tpuRtFree
 * @brief   To free device memory
 * @ingroup tpuv7_rt
 *
 * @param [out]	devPtr		memory array
 * @param [in]  free_num	free memory num
 * @retval  tpuRtSuccess  Succeeds.
 *          Other code Fails.
 */
tpuRtStatus_t tpuRtFree(void **devPtr, int free_num);

/**
 * @name    tpuRtMallocHost
 * @brief   To malloc host memory
 * @ingroup tpuv7_rt
 *
 * @param [out] ptr		memory ptr
 * @param [in]  size	memory size
 * @retval  tpuRtSuccess  Succeeds.
 *          Other code Fails.
 */
tpuRtStatus_t tpuRtMallocHost(void **ptr, unsigned long long size);

/**
 * @name    tpuRtStatus_t
 * @brief   To free host memory
 * @ingroup tpuv7_rt
 *
 * @param [in]  ptr		memory ptr
 * @param [in]  size	memory size
 * @retval  tpuRtSuccess  Succeeds.
 *          Other code Fails.
 */
tpuRtStatus_t tpuRtFreeHost(void *ptr);

/**
 * @name    tpuRtMemset
 * @brief   To set device memory
 * @ingroup tpuv7_rt
 *
 * @param [in]	ptr		memory ptr
 * @param [in]  value	memory value
 * @param [in]  size	memory size
 * @retval  tpuRtSuccess  Succeeds.
 *          Other code Fails.
 */
tpuRtStatus_t tpuRtMemset(void *devPtr, int value, unsigned long long size);

/**
 * @name    tpuRtMemcpyS2D
 * @brief   To copy date from system to device
 * @ingroup tpuv7_rt
 *
 * @param [in]	devPtr	device memory ptr
 * @param [in]  hostPtr	host memory ptr
 * @param [in]  size	memory size
 * @retval  tpuRtSuccess  Succeeds.
 *          Other code Fails.
 */
tpuRtStatus_t tpuRtMemcpyS2D(void *devPtr, const void *hostPtr, unsigned long long size);

/**
 * @name    tpuRtMemcpyD2S
 * @brief   To copy date from device to host
 * @ingroup tpuv7_rt
 *
 * @param [in]	hostPtr	host memory ptr
 * @param [in]  devPtr	device memory ptr
 * @param [in]  size	memory size
 * @retval  tpuRtSuccess  Succeeds.
 *          Other code Fails.
 */
tpuRtStatus_t tpuRtMemcpyD2S(void *hostPtr, const void *devPtr, unsigned long long size);

/**
 * @name    tpuRtMemcpyD2D
 * @brief   To copy date from device to device
 * @ingroup tpuv7_rt
 *
 * @param [in]	dstDevPtr	dst device memory ptr
 * @param [in]  srcDevPtr	src device memory ptr
 * @param [in]  size	memory size
 * @retval  tpuRtSuccess  Succeeds.
 *          Other code Fails.
 */
tpuRtStatus_t tpuRtMemcpyD2D(void *dstDevPtr, const void *srcDevPtr, unsigned long long size);
tpuRtStatus_t tpuRtMemcpyP2P(void *dstDevPtr, int dstDevice, const void *srcDevPtr, int srcDevice,
			  unsigned long long size);

/**
 * @name    tpuRtMemsetAsync
 * @brief   To set device memory asynchronously
 * @ingroup tpuv7_rt
 *
 * @param [in]	dstDevPtr	dst device memory ptr
 * @param [in]  srcDevPtr	src device memory ptr
 * @param [in]  size		memory size
 * @param [in]  stream		stream
 * @retval  tpuRtSuccess  Succeeds.
 *          Other code Fails.
 */
tpuRtStatus_t tpuRtMemsetAsync(void *devPtr, int value, unsigned long long size, tpuRtStream_t stream);

/**
 * @name    tpuRtMemcpyS2DAsync
 * @brief   To copy date from system to device asynchronously
 * @ingroup tpuv7_rt
 *
 * @param [in]	devPtr	device memory ptr
 * @param [in]  hostPtr	host memory ptr
 * @param [in]  size	memory size
 * @param [in]  stream	stream
 * @retval  tpuRtSuccess  Succeeds.
 *          Other code Fails.
 */
tpuRtStatus_t tpuRtMemcpyS2DAsync(void *devPtr, const void *hostPtr, unsigned long long size, tpuRtStream_t stream);

/**
 * @name    tpuRtMemcpyD2SAsync
 * @brief   To copy date from device to host asynchronously
 * @ingroup tpuv7_rt
 *
 * @param [in]	hostPtr	host memory ptr
 * @param [in]  devPtr	device memory ptr
 * @param [in]  size	memory size
 * @param [in]  stream	stream
 * @retval  tpuRtSuccess  Succeeds.
 *          Other code Fails.
 */
tpuRtStatus_t tpuRtMemcpyD2SAsync(void *hostPtr, const void *devPtr, unsigned long long size, tpuRtStream_t stream);

/**
 * @name    tpuRtMemcpyD2DAsync
 * @brief   To copy date from device to device asynchronously
 * @ingroup tpuv7_rt
 *
 * @param [in]	dstDevPtr	dst device memory ptr
 * @param [in]  srcDevPtr	src device memory ptr
 * @param [in]  size	memory size
 * @param [in]  stream	stream
 * @retval  tpuRtSuccess  Succeeds.
 *          Other code Fails.
 */
tpuRtStatus_t tpuRtMemcpyD2DAsync(void *dstDevPtr, const void *srcDevPtr, unsigned long long size,
				  tpuRtStream_t stream);
tpuRtStatus_t tpuRtMemcpyP2PAsync(void *dstDevPtr, int dstDevice, const void *srcDevPtr,
				  int srcDevice, unsigned long long size, tpuRtStream_t stream);

/**
 * @name    tpuRtStreamCreate
 * @brief   To create stream
 * @ingroup tpuv7_rt
 *
 * @param [out]	pStream	stream ptr
 * @retval  tpuRtSuccess  Succeeds.
 *          Other code Fails.
 */
tpuRtStatus_t tpuRtStreamCreate(tpuRtStream_t *pStream);

/**
 * @name    tpuRtStreamDestroy
 * @brief   To destory stream
 * @ingroup tpuv7_rt
 *
 * @param [in]	stream	stream ptr
 * @retval  tpuRtSuccess  Succeeds.
 *          Other code Fails.
 */
tpuRtStatus_t tpuRtStreamDestroy(tpuRtStream_t stream);

/**
 * @name    tpuRtStreamSynchronize
 * @brief   To sync stream
 * @ingroup tpuv7_rt
 *
 * @param [in]	stream	stream ptr
 * @retval  tpuRtSuccess  Succeeds.
 *          Other code Fails.
 */
tpuRtStatus_t tpuRtStreamSynchronize(tpuRtStream_t stream);

/**
 * @name    tpuRtStreamAddCallback
 * @brief   To add callback in stream
 * @ingroup tpuv7_rt
 *
 * @param [in]	stream		stream ptr
 * @param [in]	callback	function ptr
 * @param [in]	userData	function data
 * @retval  tpuRtSuccess  Succeeds.
 *          Other code Fails.
 */
tpuRtStatus_t tpuRtStreamAddCallback(tpuRtStream_t stream, pTpuRtStreamCallback callback, void *userData);

/**
 * @name    tpuRtStreamWaitEvent
 * @brief   streamA wait event in streamB
 * @ingroup tpuv7_rt
 *
 * @param [in]	stream		stream ptr
 * @param [in]	event		event in other steam
 * @retval  tpuRtSuccess  Succeeds.
 *          Other code Fails.
 */
tpuRtStatus_t tpuRtStreamWaitEvent(tpuRtStream_t stream, tpuRtEvent_t event);

/**
 * @name    tpuRtStreamWaitEventSync
 * @brief   To wait event in stream
 * @ingroup tpuv7_rt
 *
 * @param [in]	stream		stream ptr
 * @param [in]	event		event
 * @retval  tpuRtSuccess  Succeeds.
 *          Other code Fails.
 */
tpuRtStatus_t tpuRtStreamWaitEventSync(tpuRtStream_t stream, tpuRtEvent_t event);

/**
 * @name    tpuRtEventCreate
 * @brief   To create event
 * @ingroup tpuv7_rt
 *
 * @param [out]	pEvent	event ptr
 * @retval  tpuRtSuccess  Succeeds.
 *          Other code Fails.
 */
tpuRtStatus_t tpuRtEventCreate(tpuRtEvent_t *pEvent);

/**
 * @name    tpuRtEventFree
 * @brief   To free event
 * @ingroup tpuv7_rt
 *
 * @param [in]	pEvent	event ptr
 * @param [in]	stream	stream
 * @retval  tpuRtSuccess  Succeeds.
 *          Other code Fails.
 */
tpuRtStatus_t tpuRtEventFree(tpuRtEvent_t pEvent, tpuRtStream_t stream);

// tpuRtStatus_t sgEventDestroy(tpuRtEvent_t event);

/**
 * @name    tpuRtEventRecord
 * @brief   To record event in stream
 * @ingroup tpuv7_rt
 *
 * @param [in]	event	event ptr
 * @param [in]	stream	stream
 * @retval  tpuRtSuccess  Succeeds.
 *          Other code Fails.
 */
tpuRtStatus_t tpuRtEventRecord(tpuRtEvent_t event, tpuRtStream_t stream);

/**
 * @name    tpuRtEventRecord
 * @brief   To query event
 * @ingroup tpuv7_rt
 *
 * @param [in]	event	event ptr
 * @retval  tpuRtSuccess  Succeeds.
 *          Other code Fails.
 */
tpuRtStatus_t tpuRtEventQuery(tpuRtEvent_t event);

/**
 * @name    tpuRtEventSynchronize
 * @brief   To sync all event
 * @ingroup tpuv7_rt
 *
 * @param [in]	event	event ptr
 * @retval  tpuRtSuccess  Succeeds.
 *          Other code Fails.
 */
tpuRtStatus_t tpuRtEventSynchronize(tpuRtEvent_t event);
tpuRtStatus_t tpuRtEventElapsedTime(float *ms, tpuRtEvent_t start, tpuRtEvent_t end);

/**
 * @name    tpuRtKernelLoadModuleFileForCV
 * @brief   To load module for cv
 * @ingroup tpuv7_rt
 *
 * @param [in]	module_file	module path
 * @param [in]	stream		stream
 * @retval  tpuRtSuccess  Succeeds.
 *          Other code Fails.
 */
tpuRtKernelModule_t tpuRtKernelLoadModuleFileForCV(const char *module_file, tpuRtStream_t stream);

/**
 * @name    tpuRtKernelLoadModuleFile
 * @brief   To load module
 * @ingroup tpuv7_rt
 *
 * @param [in]	module_file	module path
 * @param [in]	stream		stream
 * @retval  tpuRtSuccess  Succeeds.
 *          Other code Fails.
 */
tpuRtKernelModule_t tpuRtKernelLoadModuleFile(const char *module_file, tpuRtStream_t stream);

/**
 * @name    tpuRtKernelLoadModule
 * @brief   To load module from data
 * @ingroup tpuv7_rt
 *
 * @param [in]	data	date buffer
 * @param [in]	length	date size
 * @param [in]	stream	stream
 * @retval  tpuRtSuccess  Succeeds.
 *          Other code Fails.
 */
tpuRtKernelModule_t tpuRtKernelLoadModule(const char *data, size_t length, tpuRtStream_t stream);

/**
 * @name    tpuRtKernelLoadModule
 * @brief   To launch kernel
 * @ingroup tpuv7_rt
 *
 * @param [in]	module		module ptr
 * @param [in]	func_name	function name
 * @param [in]	args		function data
 * @param [in]	size		data size
 * @param [in]	group_num	group num
 * @param [in]	block_num	block num
 * @param [in]	stream		stream
 * @retval  tpuRtSuccess  Succeeds.
 *          Other code Fails.
 */
tpuRtStatus_t tpuRtKernelLaunch(tpuRtKernelModule_t module, const char *func_name, void *args, uint32_t size,
			      uint64_t group_num, uint64_t block_num, tpuRtStream_t stream);

/**
 * @name    tpuRtKernelLaunchAsync
 * @brief   To launch kernel asynchronously
 * @ingroup tpuv7_rt
 *
 * @param [in]	module		module ptr
 * @param [in]	func_name	function name
 * @param [in]	args		function data
 * @param [in]	size		data size
 * @param [in]	group_num	group num
 * @param [in]	block_num	block num
 * @param [in]	stream		stream
 * @retval  tpuRtSuccess  Succeeds.
 *          Other code Fails.
 */
tpuRtStatus_t tpuRtKernelLaunchAsync(tpuRtKernelModule_t module, const char *func_name, void *args,
				   uint32_t size, uint64_t group_num, uint64_t block_num, tpuRtStream_t stream);

/**
 * @name    tpuRtKernelLaunchCVOP
 * @brief   To launch kernel for cvop
 * @ingroup tpuv7_rt
 *
 * @param [in]	module		module ptr
 * @param [in]	func_name	function name
 * @param [in]	args		function data
 * @param [in]	size		data size
 * @param [in]	channel_num	channel num
 * @param [in]	block_num	block num
 * @param [in]	output		output
 * @param [in]	stream		stream
 * @retval  tpuRtSuccess  Succeeds.
 *          Other code Fails.
 */
tpuRtStatus_t tpuRtKernelLaunchCVOP(tpuRtKernelModule_t module, const char *func_name, void *args, uint32_t size,
				uint32_t channel_num, uint32_t block_num,
				tpuRtLaunchOutput_t output, tpuRtStream_t stream);

/**
 * @name    tpuRtKernelLaunchCVEx
 * @brief   To launch kernel for cvex
 * @ingroup tpuv7_rt
 *
 * @param [in]	module		module ptr
 * @param [in]	func_name	function name
 * @param [in]	args		function data
 * @param [in]	size		data size
 * @param [in]	channel_num	channel num
 * @param [in]	block_num	block num
 * @param [in]	output		output
 * @param [in]	stream		stream
 * @retval  tpuRtSuccess  Succeeds.
 *          Other code Fails.
 */
tpuRtStatus_t tpuRtKernelLaunchCVEx(tpuRtKernelModule_t module, const char *func_name, void *args, uint32_t size,
				uint32_t channel_num, uint32_t block_num,
				tpuRtLaunchOutput_t output, tpuRtStream_t stream);

/**
 * @name    tpuRtKernelUnloadModule
 * @brief   To unload module
 * @ingroup tpuv7_rt
 *
 * @param [in]	p_module	module ptr
 * @param [in]	stream		stream
 * @retval  tpuRtSuccess  Succeeds.
 *          Other code Fails.
 */
tpuRtStatus_t tpuRtKernelUnloadModule(tpuRtKernelModule_t p_module, tpuRtStream_t stream);

/**
 * @name    tpuRtKernelUnloadModule
 * @brief   To get unique id
 * @ingroup tpuv7_rt
 *
 * @param [out]	uuid	unique id
 * @retval  tpuRtSuccess  Succeeds.
 *          Other code Fails.
 */
tpuRtStatus_t tpuRtGetUniqueId(char *uuid);

/**
 * @name    tpuRtKernelLaunchCDMA
 * @brief   To launch kernel cdma
 * @ingroup tpuv7_rt
 *
 * @param [in]	module		module ptr
 * @param [in]	func_name	function name
 * @param [in]	args		function data
 * @param [in]	size		data size
 * @param [in]	block_num	block num
 * @param [in]	stream		stream
 * @param [in]	cdma_only	if cdma only
 * @param [in]	uuid		unique id
 * @param [in]	rank_id		rank id
 * @param [in]	rank_num	rank num
 * @retval  tpuRtSuccess  Succeeds.
 *          Other code Fails.
 */
tpuRtStatus_t tpuRtKernelLaunchCDMA(tpuRtKernelModule_t module, const char *func_name,
				void *args, uint32_t size, uint64_t block_num, tpuRtStream_t stream,
				int cdma_only, char *uuid, int rank_id, int rank_num);

/**
 * @name    tpuRtKernelLaunchCDMA
 * @brief   To launch kernel cdma asynchronously
 * @ingroup tpuv7_rt
 *
 * @param [in]	module		module ptr
 * @param [in]	func_name	function name
 * @param [in]	args		function data
 * @param [in]	size		data size
 * @param [in]	block_num	block num
 * @param [in]	stream		stream
 * @param [in]	cdma_only	if cdma only
 * @param [in]	uuid		unique id
 * @param [in]	rank_id		rank id
 * @param [in]	rank_num	rank num
 * @retval  tpuRtSuccess  Succeeds.
 *          Other code Fails.
 */
tpuRtStatus_t tpuRtKernelLaunchCDMAAsync(tpuRtKernelModule_t module, const char *func_name,
				void *args, uint32_t size, uint64_t block_num, tpuRtStream_t stream,
				int cdma_only, char *uuid, int rank_id, int rank_num);

/**
 * @name    tpuRtSetupC2C
 * @brief   To setup c2c
 * @ingroup tpuv7_rt
 *
 * @param [in]	device_id	device id
 * @retval  tpuRtSuccess  Succeeds.
 *          Other code Fails.
 */
tpuRtStatus_t tpuRtSetupC2C(int device_id);

/**
 * @name    tpuRtSetupTopology
 * @brief   To setup Topology
 * @ingroup tpuv7_rt
 *
 * @retval  tpuRtSuccess  Succeeds.
 *          Other code Fails.
 */
tpuRtStatus_t tpuRtSetupTopology(void);

/**
 * @name    tpuRtGetTopology
 * @brief   To get topology info
 * @ingroup tpuv7_rt
 *
 * @param [out]	topology	opology info
 * @retval  tpuRtSuccess  Succeeds.
 *          Other code Fails.
 */
tpuRtStatus_t tpuRtGetTopology(struct c2c_port_info **topology);

/**
 * @name    tpuRtGetChipSN
 * @brief   To get chip sn
 * @ingroup tpuv7_rt
 *
 * @param [in]	device_id	chip id
 * @param [in]	sn			sn, should be at least 18 bytes
 * @retval  tpuRtSuccess  Succeeds.
 *          Other code Fails.
 */
tpuRtStatus_t tpuRtGetChipSN(int device_id, char *sn);

/**
 * @name    tpuRtGetPcieStatus
 * @brief   To get chip topology pcie status
 * @ingroup tpuv7_rt
 *
 * @param [in]	device_id	chip id
 * @param [in]	pcie_id		error pcie id, if there are no errors, return -1
 * @retval  tpuRtSuccess  Succeeds.
 *          Other code Fails.
 */
tpuRtStatus_t tpuRtGetPcieStatus(int device_id, int *pcie_id);
#ifdef __cplusplus
}
#endif
#endif
