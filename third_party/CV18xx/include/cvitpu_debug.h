#ifndef _CVITPU_DEBUG_H_
#define _CVITPU_DEBUG_H_

#include <stdio.h>
#include <syslog.h>
#include <assert.h>

#ifndef CVI_SUCCESS
#define CVI_SUCCESS           0
#endif
#ifndef CVI_FAILURE
#define CVI_FAILURE           -1
#endif
#define CVI_RC_SUCCESS        CVI_SUCCESS             // The operation was successful
#define CVI_RC_AGAIN          CVI_ERR_TPU_AGAIN       // Not ready yet
#define CVI_RC_FAILURE        CVI_FAILURE             // General failure
#define CVI_RC_TIMEOUT        CVI_ERR_TPU_TIMEOUT     // Timeout
#define CVI_RC_UNINIT         CVI_ERR_TPU_UNINIT      // Uninitialzed
#define CVI_RC_INVALID_ARG    CVI_ERR_TPU_INVALID_ARG // Arguments invalid
#define CVI_RC_NOMEM          CVI_ERR_TPU_NOMEM       // Not enough memory
#define CVI_RC_DATA_ERR       CVI_ERR_TPU_DATA_ERR    // Data error
#define CVI_RC_BUSY           CVI_ERR_TPU_BUSY        // Busy
#define CVI_RC_UNSUPPORT      CVI_ERR_TPU_UNSUPPORT   // Not supported yet


#if defined(__arm__) || defined(__aarch64__)
#define LOG_TOWARD_SYSLOG
#endif

#ifdef LOG_TOWARD_SYSLOG
#define TPU_LOG_FATAL(...)                  \
  do {                                      \
    syslog(LOG_LOCAL6|0, __VA_ARGS__);      \
  } while (0)

#define TPU_LOG_ERROR(...)                  \
    do {                                    \
      syslog(LOG_LOCAL6|3, __VA_ARGS__);    \
    } while (0)

#define TPU_LOG_WARNING(...)                \
    do {                                    \
      syslog(LOG_LOCAL6|4, __VA_ARGS__);    \
    } while (0)

#define TPU_LOG_NOTICE(...)                 \
    do {                                    \
      syslog(LOG_LOCAL6|5, __VA_ARGS__);    \
    } while (0)

#define TPU_LOG_INFO(...)                   \
    do {                                    \
      syslog(LOG_LOCAL6|6, __VA_ARGS__);    \
    } while (0)

#define TPU_LOG_DEBUG(...)                  \
    do {                                    \
      syslog(LOG_LOCAL6|7, __VA_ARGS__);    \
    } while (0)

#else
#define TPU_LOG_FATAL(...) printf(__VA_ARGS__)
#define TPU_LOG_ERROR(...) printf(__VA_ARGS__)
#define TPU_LOG_WARNING(...) printf(__VA_ARGS__)
#define TPU_LOG_NOTICE(...) printf(__VA_ARGS__)
#define TPU_LOG_INFO(...) printf(__VA_ARGS__)
#define TPU_LOG_DEBUG(...) printf(__VA_ARGS__)
#endif

#define NDEBUG_ASSERT
#ifdef NDEBUG_ASSERT
#define TPU_ASSERT(condition, message)                                                      \
    do {                                                                                    \
      if (!(condition)) {                                                                   \
        TPU_LOG_ERROR("%s ERROR in %s %d\n", message ? message : "", __FILE__, __LINE__);   \
        assert(0);                                                                          \
      }                                                                                     \
    } while (0)
#else
#define TPU_ASSERT(condition, message)                                                      \
    do {                                                                                    \
      assert(condition && message);                                                         \
    } while (0)
#endif

//following referened middleware pre-define
/*******************************************************************************/
/*|----------------------------------------------------------------|*/
/*| 11|   APP_ID   |   MOD_ID    | ERR_LEVEL |   ERR_ID            |*/
/*|----------------------------------------------------------------|*/
/*|<--><--6bits----><----8bits---><--3bits---><------13bits------->|*/
/*******************************************************************************/
#define CVI_TPU_ERR_APPID  (0x00000000L)
#define CVI_TPU_RUNTIME  0x77
#define CVI_TPU_ERR(module, level, errid) \
  ((int)(0xC0000000L | (CVI_TPU_ERR_APPID) | ((module) << 16) | ((level)<<13) | (errid)))

typedef enum _TPU_ERR_LEVEL_E {
  TPU_EN_ERR_LEVEL_DEBUG = 0,  /* debug-level                                  */
  TPU_EN_ERR_LEVEL_INFO,       /* informational                                */
  TPU_EN_ERR_LEVEL_NOTICE,     /* normal but significant condition             */
  TPU_EN_ERR_LEVEL_WARNING,    /* warning conditions                           */
  TPU_EN_ERR_LEVEL_ERROR,      /* error conditions                             */
  TPU_EN_ERR_LEVEL_CRIT,       /* critical conditions                          */
  TPU_EN_ERR_LEVEL_ALERT,      /* action must be taken immediately             */
  TPU_EN_ERR_LEVEL_FATAL,      /* just for compatibility with previous version */
  TPU_EN_ERR_LEVEL_BUTT
} TPU_ERR_LEVEL_E;

/* NOTE! the following defined all common error code,		*/
/*** all module must reserved 0~63 for their common error code*/
typedef enum _TPU_EN_ERR_CODE_E {
  TPU_EN_ERR_INVALID_DEVID = 1, /* invalid device ID */
  TPU_EN_ERR_INVALID_CHNID = 2, /* invalid channel ID*/
  TPU_EN_ERR_ILLEGAL_PARAM = 3,
  /* at least one parameter is illegal*/
  /* eg, an illegal enumeration value */
  TPU_EN_ERR_EXIST         = 4, /* resource exists*/
  TPU_EN_ERR_UNEXIST       = 5, /* resource unexists */
  TPU_EN_ERR_NULL_PTR      = 6, /* using a NULL point*/
  TPU_EN_ERR_NOT_CONFIG    = 7,
  /* try to enable or initialize system, device*/
  /* or channel, before configing attribute*/
  TPU_EN_ERR_NOT_SUPPORT   = 8,
  /* operation or type is not supported by NOW*/
  TPU_EN_ERR_NOT_PERM      = 9,
  /* operation is not permitted*/
  /* eg, try to change static attribute*/
  TPU_EN_ERR_INVALID_PIPEID = 10,
  /* invalid pipe ID*/
  TPU_EN_ERR_INVALID_GRPID  = 11,
  /* invalid group ID*/
  TPU_EN_ERR_NOMEM         = 12,
  /* failure caused by malloc memory*/
  TPU_EN_ERR_NOBUF         = 13,
  /* failure caused by malloc buffer*/
  TPU_EN_ERR_BUF_EMPTY     = 14,
  /* no data in buffer */
  TPU_EN_ERR_BUF_FULL      = 15,
  /* no buffer for new data*/
  TPU_EN_ERR_SYS_NOTREADY  = 16,
  /* System is not ready, maybe not initialized or*/
  /* loaded. Returning the error code when opening*/
  /* a device file failed.*/
  TPU_EN_ERR_BADADDR       = 17,
  /* bad address,*/
  /* eg. used for copy_from_user & copy_to_user*/
  TPU_EN_ERR_BUSY          = 18,
  /* resource is busy,*/
  /* eg. destroy a venc chn without unregister it */
  TPU_EN_ERR_SIZE_NOT_ENOUGH = 19,
  /* buffer size is smaller than the actual size required */
  TPU_EN_ERR_INVALID_VB    = 20,

  /* tpu error code extension */
  TPU_EN_ERR_TIMEOUT    = 21,
  TPU_EN_ERR_DATAERR    = 22,

  /* invalid VB handle */
  TPU_EN_ERR_BUTT          = 63,
  /* maximum code, private error code of all modules*/
  /* must be greater than it */
} TPU_EN_ERR_CODE_E;

typedef enum _CVI_TPU_ERRCODE {
  CVI_ERR_TPU_SUCCESS = 0,
  CVI_ERR_TPU_AGAIN = CVI_TPU_ERR(CVI_TPU_RUNTIME, TPU_EN_ERR_LEVEL_ERROR, TPU_EN_ERR_SYS_NOTREADY),
  CVI_ERR_TPU_FAILURE = -1,
  CVI_ERR_TPU_TIMEOUT = CVI_TPU_ERR(CVI_TPU_RUNTIME, TPU_EN_ERR_LEVEL_ERROR, TPU_EN_ERR_TIMEOUT),
  CVI_ERR_TPU_UNINIT = CVI_TPU_ERR(CVI_TPU_RUNTIME, TPU_EN_ERR_LEVEL_ERROR, TPU_EN_ERR_NOT_CONFIG),
  CVI_ERR_TPU_INVALID_ARG = CVI_TPU_ERR(CVI_TPU_RUNTIME, TPU_EN_ERR_LEVEL_ERROR, TPU_EN_ERR_ILLEGAL_PARAM),
  CVI_ERR_TPU_NOMEM = CVI_TPU_ERR(CVI_TPU_RUNTIME, TPU_EN_ERR_LEVEL_ERROR, TPU_EN_ERR_NOMEM),
  CVI_ERR_TPU_DATA_ERR = CVI_TPU_ERR(CVI_TPU_RUNTIME, TPU_EN_ERR_LEVEL_ERROR, TPU_EN_ERR_DATAERR),
  CVI_ERR_TPU_BUSY = CVI_TPU_ERR(CVI_TPU_RUNTIME, TPU_EN_ERR_LEVEL_ERROR, TPU_EN_ERR_BUSY),
  CVI_ERR_TPU_UNSUPPORT = CVI_TPU_ERR(CVI_TPU_RUNTIME, TPU_EN_ERR_LEVEL_ERROR, TPU_EN_ERR_NOT_SUPPORT),
} CVI_TPU_ERRCODE;

#endif
