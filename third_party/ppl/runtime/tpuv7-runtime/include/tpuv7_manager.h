/* SPDX-License-Identifier: GPL-2.0 */

#ifndef __TPUV7_M__
#define __TPUV7_M__

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tpuRt_misc_info {
	char domain_bdf[64];
	char driver_version[64];
	char chipid[16];
} tpuRt_misc_info_t;

typedef struct dev_stat {
	long long Mem_total;
	long long Mem_used;
	int Tpu_util;
} dev_stat_t;

/**
 * @name    tpuRtGetChipCount
 * @brief   To get chip count
 * @ingroup tpuv7_manager
 *
 * @param [in]	chip_count  chip count
 * @retval  tpuRtSuccess  Succeeds.
 *          Other code Fails.
 */
tpuRtStatus_t tpuRtGetChipCount(int *chip_count);

/**
 * @name    tpuRtGetStat
 * @brief   To get chip status
 * @ingroup tpuv7_manager
 *
 * @param [in]	device_id	device id
 * @param [in]	stat		chip status
 * @retval  tpuRtSuccess  Succeeds.
 *          Other code Fails.
 */
tpuRtStatus_t tpuRtGetStat(int device_id, dev_stat_t *stat);

/**
 * @name    tpuRtGetMiscInfo
 * @brief   To get misc info
 * @ingroup tpuv7_manager
 *
 * @param [in]	device_id	device id
 * @param [in]	misc		misc info
 * @retval  tpuRtSuccess  Succeeds.
 *          Other code Fails.
 */
tpuRtStatus_t tpuRtGetMiscInfo(int device_id, tpuRt_misc_info_t *misc);

/**
 * @name    tpuRtGetBoardMaxPower
 * @brief   To get board max power
 * @ingroup tpuv7_manager
 *
 * @param [in]	device_id	device id
 * @param [in]	board_max_power	board max power
 * @retval  tpuRtSuccess  Succeeds.
 *          Other code Fails.
 */
tpuRtStatus_t tpuRtGetBoardMaxPower(int device_id, int *board_max_power);

/**
 * @name    tpuRtGetBoardPower
 * @brief   To get board power
 * @ingroup tpuv7_manager
 *
 * @param [in]	device_id	device id
 * @param [in]	board_power	board power
 * @retval  tpuRtSuccess  Succeeds.
 *          Other code Fails.
 */
tpuRtStatus_t tpuRtGetBoardPower(int device_id, int *board_power);

/**
 * @name    tpuRtGetBoardTemp
 * @brief   To get board temperature
 * @ingroup tpuv7_manager
 *
 * @param [in]	device_id	device id
 * @param [in]	board_temp	board temperature
 * @retval  tpuRtSuccess  Succeeds.
 *          Other code Fails.
 */
tpuRtStatus_t tpuRtGetBoardTemp(int device_id, int *board_temp);

/**
 * @name    tpuRtGetChipTemp
 * @brief   To get chip temperature
 * @ingroup tpuv7_manager
 *
 * @param [in]	device_id	device id
 * @param [in]	chip_temp	chip temperature
 * @retval  tpuRtSuccess  Succeeds.
 *          Other code Fails.
 */
tpuRtStatus_t tpuRtGetChipTemp(int device_id, int *chip_temp);

/**
 * @name    tpuRtGetChipSN
 * @brief   To get chip sn
 * @ingroup tpuv7_manager
 *
 * @param [in]	device_id	chip id
 * @param [in]	sn			sn, should be at least 18 bytes
 * @retval  tpuRtSuccess  Succeeds.
 *          Other code Fails.
 */
tpuRtStatus_t tpuRtGetSN(int device_id, char *sn);

/**
 * @name    tpuRtGetBoardName
 * @brief   To get chip temperature
 * @ingroup tpuv7_manager
 *
 * @param [in]	device_id	device id
 * @param [in]	board_name	board name
 * @retval  tpuRtSuccess  Succeeds.
 *          Other code Fails.
 */
tpuRtStatus_t tpuRtGetBoardName(int device_id, char *board_name);

/**
 * @name    tpuRtGetTpuGoodNumber
 * @brief   To get good tpu number
 * @ingroup tpuv7_manager
 *
 * @param [in]	device_id	device id
 * @param [in]	tpu_good_number	good tpu number
 * @retval  tpuRtSuccess  Succeeds.
 *          Other code Fails.
 */
tpuRtStatus_t tpuRtGetTpuGoodNumber(int device_id, int *tpu_good_number);

#ifdef __cplusplus
}
#endif
#endif
