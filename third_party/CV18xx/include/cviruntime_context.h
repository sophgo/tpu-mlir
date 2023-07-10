#ifndef _CVIRUNTIME_CONTEXT_H_
#define _CVIRUNTIME_CONTEXT_H_

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include "cvitpu_debug.h"

#ifdef __cplusplus
  extern "C" {
#endif

typedef void *CVI_RT_HANDLE;
typedef void *CVI_RT_SHANDLE;
typedef void *CVI_RT_KHANDLE;
typedef void *CVI_RT_MEM;
typedef int CVI_RC;

typedef struct __CVI_RT_ARRAYBASE {
  uint64_t gaddr_base0;
  uint64_t gaddr_base1;
  uint64_t gaddr_base2;
  uint64_t gaddr_base3;
  uint64_t gaddr_base4;
  uint64_t gaddr_base5;
  uint64_t gaddr_base6;
  uint64_t gaddr_base7;
} CVI_RT_ARRAYBASE;

typedef enum {
  CVI_ALLOC_WEIGHT = 0,
  CVI_ALLOC_PROGRAM = 1,
  CVI_ALLOC_NEURON = 2,
  CVI_ALLOC_SHARED = 3,
  CVI_ALLOC_DMABUF = 4,
  CVI_ALLOC_UNKNOWN = 5
} CVI_ALLOC_TYPE;

typedef CVI_RT_MEM (*CVI_MEM_ALLOC_CB) (CVI_RT_HANDLE, uint64_t, CVI_ALLOC_TYPE, const char *);
typedef void (*CVI_MEM_FREE_CB) (CVI_RT_HANDLE, CVI_RT_MEM);

CVI_RC CVI_RT_Init(CVI_RT_HANDLE *rt_handle);
CVI_RC CVI_RT_DeInit(CVI_RT_HANDLE rt_handle);

CVI_RT_KHANDLE CVI_RT_RegisterKernel(CVI_RT_HANDLE rt_handle, uint32_t cmdbuf_size);
CVI_RC CVI_RT_UnRegisterKernel(CVI_RT_KHANDLE rt_khandle);

CVI_RC CVI_RT_Submit(CVI_RT_KHANDLE rt_khandle);
CVI_RC CVI_RT_SubmitAsync(CVI_RT_KHANDLE rt_khandle, uint8_t submit_previous);
CVI_RC CVI_RT_WaitForAsync(CVI_RT_KHANDLE rt_khandle);

CVI_RC CVI_RT_LoadCmdbuf(
    CVI_RT_HANDLE rt_handle, uint8_t *cmdbuf,
    uint64_t cmdbuf_sz, uint64_t gaddr_base0,
    uint64_t gaddr_base1, bool enable_pmu,
    CVI_RT_MEM *cmdbuf_mem);
CVI_RC CVI_RT_LoadDmabuf(
    CVI_RT_HANDLE rt_handle, CVI_RT_MEM dmabuf,
    uint64_t cmdbuf_sz, uint64_t gaddr_base0,
    uint64_t gaddr_base1, bool enable_pmu, CVI_RT_MEM *dmabuf_mem);
CVI_RC CVI_RT_RunCmdbuf(
    CVI_RT_HANDLE rt_handle, CVI_RT_MEM cmdbuf_mem,
    uint64_t gaddr_base2, uint64_t gaddr_base3);
CVI_RC CVI_RT_RunCmdbufEx(
    CVI_RT_HANDLE rt_handle, CVI_RT_MEM cmdbuf_mem,
    CVI_RT_ARRAYBASE *p_array_base);

CVI_RC CVI_RT_LoadCmdbufTee(
    CVI_RT_HANDLE rt_handle, uint8_t *cmdbuf,
    size_t sz, uint64_t neuron_gaddr, uint64_t weight_gaddr,
    uint32_t weight_len, CVI_RT_MEM *cmdbuf_mem);

CVI_RC CVI_RT_RunCmdbufTee(
    CVI_RT_HANDLE rt_handle, CVI_RT_MEM cmdbuf_mem,
    CVI_RT_ARRAYBASE *p_array_base);

CVI_RT_MEM CVI_RT_MemAlloc(CVI_RT_HANDLE rt_handle, uint64_t size);
CVI_RT_MEM CVI_RT_MemPreAlloc(CVI_RT_MEM mem, uint64_t offset, uint64_t size);
void CVI_RT_MemFree(CVI_RT_HANDLE rt_handle, CVI_RT_MEM mem);
void CVI_RT_MemFreeEx(uint64_t p_addr);
uint64_t CVI_RT_MemGetSize(CVI_RT_MEM mem);
uint64_t CVI_RT_MemGetPAddr(CVI_RT_MEM mem);
uint8_t* CVI_RT_MemGetVAddr(CVI_RT_MEM mem);
int32_t CVI_RT_MemIncRef(CVI_RT_MEM mem);
int32_t CVI_RT_MemDecRef(CVI_RT_MEM mem);

CVI_RC CVI_RT_MemCopyS2D(CVI_RT_HANDLE rt_handle, CVI_RT_MEM dst, uint8_t* src);
CVI_RC CVI_RT_MemCopyD2S(CVI_RT_HANDLE rt_handle, uint8_t* dst, CVI_RT_MEM src);
CVI_RC CVI_RT_MemCopyS2DEx(
    CVI_RT_HANDLE rt_handle, CVI_RT_MEM dst,
    uint64_t offset, uint64_t len, uint8_t* src);
CVI_RC CVI_RT_MemFlush(CVI_RT_HANDLE rt_handle, CVI_RT_MEM mem);
CVI_RC CVI_RT_MemInvld(CVI_RT_HANDLE rt_handle, CVI_RT_MEM mem);
CVI_RC CVI_RT_MemFlushEx(CVI_RT_HANDLE rt_handle, CVI_RT_MEM mem, uint64_t len);
CVI_RC CVI_RT_MemInvldEx(CVI_RT_HANDLE rt_handle, CVI_RT_MEM mem, uint64_t len);

CVI_RC CVI_RT_ParsePmuBuf(CVI_RT_MEM cmdbuf_mem, uint8_t **buf_start, uint32_t *buf_len);

CVI_RC CVI_RT_SetBaseReg(CVI_RT_HANDLE rt_handle, uint32_t inx, uint64_t base_addr);

/*
 * set memory alloc and free callback function.
 * @param [in] CVI_MEM_ALLOC_CB,  memory alloc function
 * @param [in] CVI_MEM_FREE_CB,  memory free function
 */
CVI_RC CVI_RT_Global_SetMemAllocCallback(
    CVI_MEM_ALLOC_CB alloc_cb, CVI_MEM_FREE_CB free_cb);

/*
 * reset to default memory alloc and free function.
 */
void CVI_RT_Global_ResetMemAllocCallback();

#ifdef __cplusplus
}
#endif

#endif // _CVIRUNTIME_CONTEXT_H_

