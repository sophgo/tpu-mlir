//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx_profiling.hpp"
#include "tpu_mlir/Backend/CV18xx/CV18xx.h"
#include "tpu_mlir/Support/MathUtils.h"

namespace tpu_mlir {
namespace backend {
static inline uint64_t align_down(uint64_t x, uint64_t n) { return x / n * n; }

uint64_t CV18xxProfiling::get_cycle(std::vector<uint8_t> &cmdbuf) {
  int64_t offset = 0;
  int64_t tiuCnt = 0;
  int64_t tdmaCnt = 0;
  tiu_reg_t tiuReg;
  tdma_reg_t tdmaReg;
  std::vector<tiu_reg_t> tiuCmdList;
  std::vector<tdma_reg_t> tdmaCmdList;

  int64_t cmdBufSize = cmdbuf.size();
  unsigned char *pContent = nullptr;
  unsigned char *pCurrent = (unsigned char *)cmdbuf.data();
  cmd_hdr_t *pHeader = nullptr;

  uint64_t tpu_freqency = 0;
  uint64_t dram_freqency = 1886;
  bool assign_tpu_freqency = false;
  while (offset < cmdBufSize) {
    int tdmaRound65565Times = tdmaCnt / 0xffff;
    int tiuRound65565Times = tiuCnt / 0xffff;
    pHeader = (cmd_hdr_t *)pCurrent;
    // printf("engine_id=%d\n", pHeader->engine_id);
    pContent = pCurrent + sizeof(cmd_hdr_t);

    if (assign_tpu_freqency == false) {
      assign_tpu_freqency = true;
      switch (pHeader->magic) {
      case MAGIC_CV183X:
        tpu_freqency = 650;
        break;
      case MAGIC_CV182X:
        tpu_freqency = 750;
        break;
      case MAGIC_CV181X:
      case MAGIC_CV180X:
        tpu_freqency = 500;
        break;
      default:
        assert(0 && "unsupported chip");
        break;
      }
    }

    if (pHeader->engine_id == 0) {
      TiuReg::parse_tiu_reg(&tiuReg, (uint32_t *)pContent, pHeader->magic);
      if (tiuReg.cmd_id_gdma != 0) {
        tiuReg.cmd_id_gdma += 65535 * tdmaRound65565Times;
      }
      if (tiuReg.cmd_id_tpu != 0) {
        tiuReg.cmd_id_tpu += 65535 * tiuRound65565Times;
      }
      tiuCmdList.emplace_back(std::move(tiuReg));
      tiuCnt++;
    } else if (pHeader->engine_id == 2) {
      TdmaReg::parse_tdma_reg(&tdmaReg, (uint32_t *)pContent, pHeader->magic);
      if (tdmaReg.cmd_id != 0) {
        tdmaReg.cmd_id += 65535 * tdmaRound65565Times;
      }
      if (tdmaReg.wait_id_tpu != 0) {
        tdmaReg.wait_id_tpu += 65535 * tiuRound65565Times;
      }
      tdmaCmdList.emplace_back(std::move(tdmaReg));
      tdmaCnt++;
    }

    // ppmu_cmdbuf_fillinfo(pHeader->engine_id, pContent);

    pCurrent = pContent + pHeader->len;
    offset += sizeof(cmd_hdr_t) + pHeader->len;
  }
  // printf("tdmaCnt=%ld, tiuCnt=%ld\n", tdmaCnt, tiuCnt);
  uint64_t tiu_cycle = 0;
  uint64_t tdma_cycle = 0;
  for (auto task : tiuCmdList) {
    tiu_cycle += TiuReg::calCycle(task, tpu_freqency);
  }
  for (auto task : tdmaCmdList) {
    tdma_cycle += TdmaReg::calCycle(task, dram_freqency, tpu_freqency);
  }
  // printf("tiu_cycle=%ld, tdma_cycle=%ld\n", tiu_cycle, tdma_cycle);
  return tiu_cycle + tdma_cycle;
}

uint64_t TiuReg::calCycle(tiu_reg_t &task, uint64_t tpu_frequency) {
  uint64_t cycle_count = calTiuCycle(task);
  uint64_t total_time = cycle_count * 1000 / tpu_frequency;
  return total_time;
}

void TiuReg::parse_tiu_reg(tiu_reg_t *r, const uint32_t *p,
                           unsigned char magicNum) {
  if (magicNum == 0xA5)
    parse_cv183x_tiu_reg(r, p);
  else if (magicNum == 0xA6) {
    parse_cv182x_tiu_reg(r, p);
    cv182xMapToCv183x(r);
  } else if (magicNum == 0xA7) {
    parse_cv182x_tiu_reg(r, p);
    cv182xMapToCv183x(r);
  } else if (magicNum == 0xA8) {
    parse_cv182x_tiu_reg(r, p);
    cv182xMapToCv183x(r);
  } else {
    printf("invalid magic number %x\n", magicNum);
    assert(0);
  }
}

uint64_t TiuReg::calTiuCycle(tiu_reg_t &task) {
  Des_tsk_typ desTskType = static_cast<Des_tsk_typ>(task.tsk_typ);
  uint64_t tempCycle = 0;
  uint64_t resCStart =
      (task.res0_addr - LOCAL_MEM_START_ADDR) >> TPU_LMEM_ADDR_WIDTH;
  Tuple4D kernelShape =
      Tuple4D(task.opd1_n, task.opd1_h, task.opd1_w, task.opd1_c);
  Tuple4D activationInputShape =
      Tuple4D(task.opd0_n, task.opd0_h, task.opd0_w, task.opd0_c);
  // Tuple4D activationStride = Tuple4D(task.opd0_n_str, task.opd0_h_str,
  // task.opd0_w_str, task.opd0_c_str);
  Tuple4D activationOutputShape =
      Tuple4D(task.res0_n, task.res0_h, task.res0_w, task.res0_c);
  int pSumModeLat = (task.ps32_md == 0)   ? 0
                    : (task.ps32_md == 1) ? 4
                    : (task.ps32_md == 2) ? 4
                                          : 8; // partial R/W needs 4 cycle
  const int syncR0R1PathCycle =
      37 + 2 + 2; // 37T : 2Array, 37 + 2 + 2 = 4array BDC + cmd latency
  const int cmdLatency =
      15 + 8 + 1 + 2; // Without broadcast initLat + broadcast lane 24T :
                      // 2Array, 24 + 2 = 4array (cmd latency
  float misc;         // magic ratio every mac
  bool isPerChannelQuan = (task.opt_chl_quan == 1);
  bool isBfloat16 = (task.opd_typ == 1);
  int perLaneEuNumber = CV18xx::EU_BYTES;
  int perChannelQuanLeastCycle = perLaneEuNumber; // Todo : confirm this
  const int postProcessCycle =
      (isPerChannelQuan) ? 31 : 12; // 18T EU 16 + 1 + 1
  // Mac(kh * kw * ic) should be larger than this cycle, or bubble will appear
  uint64_t activatedEuNumber =
      (isBfloat16) ? perLaneEuNumber / 2 : perLaneEuNumber;
  int biasLat = (task.tsk_opd_num == 3) ? (isPerChannelQuan) ? 0 : 2
                                        : 0; // load 16 bit data need 2 cycle
  int shiftRoundShiftCycle = isBfloat16 ? 2 : (isPerChannelQuan) ? 1 : 4;
  uint64_t laneNumber = CV18xx::NPU_NUM;

  switch (desTskType) {
  case (Conv2D): {
    // ToDo : Analyze inputStride effect
    // ToDo : partial sum

    uint64_t channelNumPerCyc =
        (task.double_conv == 1) ? (!isBfloat16) ? 2 : 1 : 1;
    // int fakeIc = (activationInputShape.c >= 4) ? activationInputShape.c :
    // 4;
    // After mac  w/   bais : Hazard(1) | ADDDB (2) | SHIFTR (1) | ADDSD (1) |
    // SHIFTD(1) After mac  w/o  bais : Hazard(1) |             SHIFTR (1) |
    // ADDSD (1) | SHIFTD(1)

    int loopCycle =
        int(ceil((kernelShape.w * kernelShape.h *
                  ceiling_func(activationInputShape.c, channelNumPerCyc))) +
            biasLat + shiftRoundShiftCycle + pSumModeLat);
    // if(activationInputShape.c < 4) {
    //     int bubble = (kernelShape.w * kernelShape.h == 1) ? 0 :
    //     ((((int)(kernelShape.w * kernelShape.h / 2) - 1) * (4 -
    //     activationInputShape.c)));
    uint32_t fetchCyclePerTime =
        ceil((((float)(perLaneEuNumber - 1) * task.conv_op_x_str) /
                  (task.conv_opd0_x_ins0 + 1) +
              1) /
             perLaneEuNumber) +
        1;
    int bubble = (fetchCyclePerTime > activationInputShape.c)
                     ? ceil(kernelShape.w * kernelShape.h) *
                           (fetchCyclePerTime - activationInputShape.c)
                     : 0;
    // cout << "fetchCyclePerTime = " << fetchCyclePerTime << endl;
    // cout << "bubble = " << bubble << endl;
    loopCycle =
        int(ceil((kernelShape.w * kernelShape.h *
                      ceiling_func(activationInputShape.c, channelNumPerCyc) +
                  bubble)) +
            biasLat + shiftRoundShiftCycle +
            pSumModeLat); // Another bubble ~= 25 cycle
    loopCycle = (isPerChannelQuan && !isBfloat16)
                    ? (loopCycle > perChannelQuanLeastCycle)
                          ? loopCycle
                          : perChannelQuanLeastCycle
                    : loopCycle;
    // every ic * 2 get 4 - ic bubble
    // ic = 1 : ic + (ic/2 * 4-ic)
    // }

    tempCycle = syncR0R1PathCycle;
    tempCycle += activationOutputShape.n *
                 ceiling_func(resCStart + activationOutputShape.c, laneNumber) *
                 ceiling_func(activationOutputShape.w * activationOutputShape.h,
                              activatedEuNumber) *
                 loopCycle;
    // Consider inputC instead of opd1C because of definition
    tempCycle += postProcessCycle;
    break;
  }
  case (Pooling): {
    // 0 -> max pooling
    // 1 -> avg pooling
    // 2 -> depthwise

    // ToDo : Analyze inputStride effect
    switch (task.tsk_eu_typ) {
    case 0: // max pooling
    {
      // Todo : Analyze output.w <= 16

      int fakeOw = perLaneEuNumber;
      misc = (((float)(fakeOw - 1) * task.conv_op_x_str) /
                  (task.conv_opd0_x_ins0 + 1) +
              1) /
             perLaneEuNumber; // Consider small OW
      misc += (((float)(fakeOw - 1) * task.conv_op_x_str) /
                   (task.conv_opd0_x_ins0 + 1) +
               1) >= perLaneEuNumber
                  ? ((float)perLaneEuNumber - 1) / perLaneEuNumber
                  : ((((float)(fakeOw - 1) * task.conv_op_x_str) /
                          (task.conv_opd0_x_ins0 + 1) +
                      1) -
                     1) /
                        perLaneEuNumber; // Consider inst > stride
      misc = misc < 1 ? 1 : misc;
      misc =
          ((activationOutputShape.w * activationOutputShape.h == 1)) ? 1 : misc;
      const int magicHazard = 2;
      // This magicHazard appears when data gathers need < 2T
      // cout << "misc = " << misc << endl;
      tempCycle =
          activationOutputShape.n *
          ceiling_func((resCStart + activationOutputShape.c), laneNumber) *
          ceiling_func(activationOutputShape.w * activationOutputShape.h,
                       activatedEuNumber) *
          (kernelShape.h * kernelShape.w * misc + magicHazard);
      break;
    }
    case 1: // avg pooling
    {
      // Todo : Analyze output.w <= 16
      int fakeOw = activatedEuNumber;
      // int fakeOw = activationOutputShape.w >= activatedEuNumber ?
      // activatedEuNumber : activationOutputShape.w;
      misc = (((float)(fakeOw - 1) * task.conv_op_x_str) /
                  (task.conv_opd0_x_ins0 + 1) +
              1) /
             activatedEuNumber; // Consider small OW
      misc += (((float)(fakeOw - 1) * task.conv_op_x_str) /
                   (task.conv_opd0_x_ins0 + 1) +
               1) >= activatedEuNumber
                  ? ((float)activatedEuNumber - 1) / activatedEuNumber
                  : ((((float)(fakeOw - 1) * task.conv_op_x_str) /
                          (task.conv_opd0_x_ins0 + 1) +
                      1) -
                     1) /
                        activatedEuNumber; // Consider inst > stride
      // cout << misc  << endl;
      misc = misc < 1 ? 1 : misc;
      misc =
          ((activationOutputShape.w * activationOutputShape.h == 1)) ? 1 : misc;
      int loopCycle = kernelShape.h * kernelShape.w * misc;
      loopCycle = (isPerChannelQuan) ? (loopCycle > perChannelQuanLeastCycle)
                                           ? loopCycle
                                           : perChannelQuanLeastCycle
                                     : loopCycle;
      tempCycle =
          activationOutputShape.n *
          ceiling_func((resCStart + activationOutputShape.c), laneNumber) *
          ceiling_func(activationOutputShape.w * activationOutputShape.h,
                       activatedEuNumber) *
          (loopCycle + shiftRoundShiftCycle);
      tempCycle += postProcessCycle;
      /* test code
      Tuple4D startPoint = Tuple4D(0, 0, 0, 0);
      Tuple4D endPoint = Tuple4D(0, activationOutputShape.h,
      activationOutputShape.w, 0); uint32_t sramAddr[perLaneEuNumber],
      preSramAddr; int cnt; int insX = task.conv_opd0_x_ins0 + 1; int insY =
      task.conv_opd0_x_ins0 + 1; uint64_t dataFetchCycle = 0;

      // cout << "endPoint = " << endPoint <<endl;
      while((startPoint.h <= endPoint.h) && (startPoint.w <= endPoint.w)) {
          for(int kwCounter = 0; kwCounter < kernelShape.w; kwCounter++) {
              for(int khCounter = 0; khCounter < kernelShape.h; khCounter++) {
                  cnt = 1;
                  for(int i = 0; i < perLaneEuNumber; i++) {
                      int tmpOutputWidth = startPoint.w + i;
                      int tmpOutputHeight = startPoint.h + tmpOutputWidth /
      activationOutputShape.w; tmpOutputWidth %= activationOutputShape.w;
                      sramAddr[i] = (task.opd0_addr + (tmpOutputWidth *
      task.conv_op_x_str + kwCounter) + (tmpOutputHeight * task.conv_op_y_str
      + khCounter) * activationOutputShape.w) / 16;

                      if(((tmpOutputWidth *  task.conv_op_x_str + kwCounter) %
      insX) != 0) { sramAddr[i] = 0xdeadbeaf;
                      }

                      if(((tmpOutputHeight * task.conv_op_y_str + khCounter) %
      insY) != 0) { sramAddr[i] = 0xdeadbeaf;
                      }
                      // cout << "tmpOutputWidth = " << tmpOutputWidth <<
      endl;
                      // cout << "tmpOutputHeight = " << tmpOutputHeight <<
      endl;
                      // cout << "sramAddr[i] = " << sramAddr[i] << endl;
                      // cout << "byte = " << (task.opd0_addr +
      (tmpOutputWidth *  task.conv_op_x_str + kwCounter) + (tmpOutputHeight *
      task.conv_op_y_str + khCounter) * activationOutputShape.w) % 16 << endl;
                  }
                  preSramAddr = sramAddr[0];
                  for(int i = 1; i < perLaneEuNumber; i++) {
                      if(sramAddr[i] == 0xdeadbeaf) {
                          continue;
                      }
                      if(sramAddr[i] != preSramAddr) {
                          cnt++;
                          preSramAddr = sramAddr[i];
                      }
                  }
                  // cout << "cnt = " << cnt << endl;
                  dataFetchCycle += cnt;
                  // cout << endl;
              }
          }
          dataFetchCycle += shiftRoundShiftCycle;
          startPoint.w += perLaneEuNumber;
          startPoint.h += startPoint.w / activationOutputShape.w;
          startPoint.w %= activationOutputShape.w;
          // cout << startPoint << endl;
      }
      tempCycle =
          activationOutputShape.n *
          ceiling_func((resCStart + activationOutputShape.c), laneNumber) *
          dataFetchCycle;
      */
      break;
    }
    case 2: // depthwise
    {
      // Too much bubble
      // ToDo : Analyze act+wt bank conflict
      // Todo : Analyze output.w <= 16
      shiftRoundShiftCycle =
          (((kernelShape.w * kernelShape.h == 1) && (!isPerChannelQuan)) ||
           (!isBfloat16))
              ? 3
              : shiftRoundShiftCycle;

      int fakeOw = (int)activationOutputShape.w >= perLaneEuNumber
                       ? perLaneEuNumber
                       : activationOutputShape.w;
      float ohNumberOneTime =
          (int)activationOutputShape.w >= perLaneEuNumber
              ? 1
              : (float)perLaneEuNumber / activationOutputShape.w;
      misc = (((float)(fakeOw - 1) * task.conv_op_x_str) /
                  (task.conv_opd0_x_ins0 + 1) +
              1) /
             perLaneEuNumber * ohNumberOneTime; // Consider small OW
      misc += ((float)perLaneEuNumber - 1) / perLaneEuNumber;
      misc = ((kernelShape.w * kernelShape.h == 1)) ? 1 : misc;
      int loopCycle = kernelShape.h * kernelShape.w * misc;
      loopCycle = (isPerChannelQuan) ? (loopCycle > perChannelQuanLeastCycle)
                                           ? loopCycle
                                           : perChannelQuanLeastCycle
                                     : loopCycle;

      tempCycle =
          activationOutputShape.n *
          ceiling_func((resCStart + activationOutputShape.c), laneNumber) *
          ceiling_func(activationOutputShape.w * activationOutputShape.h,
                       activatedEuNumber) *
          (loopCycle + biasLat + shiftRoundShiftCycle);
      tempCycle += postProcessCycle;
      // tempCycle = misc * tempCycle;
      break;
    }
    default:
      assert(false);
    }
    break;
  }
  case (MatrixMul): {
    // ToDo : Analyze act+wt bank conflict
    uint32_t addResLat = (task.opt_res_add) ? 2 : 0; // 16bit

    tempCycle = syncR0R1PathCycle;
    tempCycle += activationOutputShape.n *
                 ceiling_func(activationOutputShape.c, laneNumber) *
                 ceiling_func(activationOutputShape.w, activatedEuNumber) *
                 (kernelShape.n + biasLat + shiftRoundShiftCycle + pSumModeLat +
                  addResLat);
    tempCycle += postProcessCycle;
    break;
  }
  case (TensorArithmetic): {
    bool isInput8BitMode = (task.opt_opd0_int8 == 1);
    bool isRes8BitMode = (task.opt_res0_int8 == 1);
    bool isOpd1Const = (task.opt_opd1_const == 1);
    // mode 0 : mul res8bit/add/sub
    // mode 1 : mac
    // mode 2 : max/min/shift/logic/mul res16bit
    // mode 3 : mdsum
    // mode 4 : lut
    int mode = getTensorArithmeticMode(task.tsk_eu_typ, isRes8BitMode);
    // const int shiftRoundShiftCycle = (mode == 0) ? 3 : 0;

    float euLat =
        ((task.tens_lookup == 1))
            ? 0
            : getEltwiseLatency(task.tsk_eu_typ, isInput8BitMode, isOpd1Const,
                                mode); // getEltwiseLatency(int taskType, bool
                                       // isInput8BitMode)
    euLat += isRes8BitMode ? 0 : 1;
    float writeLat = (task.res0_c_str % 16 == 0)
                         ? 1
                         : (float)30 / 16; // consider stride more
    tempCycle = activationOutputShape.n *
                ceiling_func(resCStart + activationOutputShape.c, laneNumber) *
                ceiling_func(activationOutputShape.w * activationOutputShape.h,
                             activatedEuNumber) *
                (euLat * writeLat);

    if (task.tens_lookup == 1) {
      tempCycle =
          activationOutputShape.n *
          ceiling_func(resCStart + activationOutputShape.c, laneNumber) *
          ceiling_func(activationOutputShape.w * activationOutputShape.h,
                       activatedEuNumber) *
          (1 + task.opd1_h + 2); // opa, opb, bubble
    }

    if (task.tens_mdsum == 1) {
      tempCycle =
          activationOutputShape.n *
          ceiling_func(resCStart + activationOutputShape.c, laneNumber) *
          ceiling_func(activationOutputShape.w * activationOutputShape.h,
                       activatedEuNumber) *
          (1 + 2); // opa, bubble + latency
    }

    tempCycle += cmdLatency;
    tempCycle += postProcessCycle;
    break;
  }
  case (MatrixMul2): {
    std::cout << "Not supported now" << std::endl;
    assert(0);
    break;
  }
  default: {
    std::cout << "Not supported now" << std::endl;
    assert(0);
    break;
  }
  }
  return tempCycle;
}

float TiuReg::getEltwiseLatency(int taskType, bool is8BitMode, bool isOpd1Const,
                                int mode) {
  float ret;
  if (taskType == 0) { // Todo : random bubble, HW will erase it 2.5->2
    // mul
    ret = (is8BitMode) ? 2.5 : 4.5; // 8bit : rounding shift
    ret += (mode == 0) ? 3 : 0;
    // ret -= (isOpd1Const) ? 1 : 0;
  } else if (taskType == 1) {
    // mac
    ret = (is8BitMode) ? 9 : 11;
  } else if (taskType == 2) {
    // add
    ret = (is8BitMode) ? 5 : 7;
    ret -= (!isOpd1Const) ? 0 : (is8BitMode) ? 0 : 0.5; // remove bank conflic
  } else if (taskType == 3) {
    // sub
    ret = (is8BitMode) ? 5 : 7;
    ret -= (!isOpd1Const) ? 0 : (is8BitMode) ? 0 : 0.5; // remove bank conflic
  } else if (taskType == 4) {
    // max
    ret = (is8BitMode) ? 3 : 5;
  } else if (taskType == 5) {
    // min
    ret = (is8BitMode) ? 3 : 5;
  } else if (taskType == 6) {
    // shift
    ret = (is8BitMode) ? 4 : 4;
  } else if (taskType == 7) {
    // and
    ret = (is8BitMode) ? 3 : 5;
  } else if (taskType == 8) {
    // or
    ret = (is8BitMode) ? 3 : 5;
  } else if (taskType == 9) {
    // xor
    ret = (is8BitMode) ? 3 : 5;
  } else if (taskType == 10) {
    // copy
    ret = (is8BitMode) ? 2 : 4;
  } else if (taskType == 11) {
    // ge
    ret = (is8BitMode) ? 3 : 5;
  } else if (taskType == 12) {
    std::cout << "Not supported now" << std::endl;
    assert(0);
  } else {
    std::cout << "Not supported now" << std::endl;
    assert(0);
  }
  return ret;
}

int TiuReg::getTensorArithmeticMode(int taskType, bool is8BitMode) {
  // mode 0 : mul res8bit/add/sub
  // mode 1 : mac
  // mode 2 : max/min/shift/logic/mul res16bit
  // mode 3 : mdsum
  // mode 4 : lut
  int ret;
  if (taskType == 0) {
    // mul
    ret = (is8BitMode) ? 0 : 2;
  } else if (taskType == 1) {
    // mac
    ret = 1;
  } else if (taskType == 2) {
    // add
    ret = 0;
  } else if (taskType == 3) {
    // sub
    ret = 0;
  } else if (taskType == 4) {
    // max
    ret = 2;
  } else if (taskType == 5) {
    // min
    ret = 2;
  } else if (taskType == 6) {
    // shift
    ret = 2;
  } else if (taskType == 7) {
    // and
    ret = 2;
  } else if (taskType == 8) {
    // or
    ret = 2;
  } else if (taskType == 9) {
    // xor
    ret = 2;
  } else if (taskType == 10) {
    // copy
    ret = 2;
  } else if (taskType == 11) {
    // md_sum
    ret = 3;
  } else if (taskType == 12) {
    // lut
    ret = 4;
  } else {
    std::cout << "Not supported now" << std::endl;
    assert(0);
  }
  return ret;
}
void TiuReg::cv182xMapToCv183x(tiu_reg_t *r) {
  bool isRes_16Bit = r->opt_res0_seg == 0 ? true : false;
  //   bool isOpd0_16Bit = r->opt_opd0_seg == 0 ? true : false;
  //   bool isOpd1_16Bit = r->opt_opd1_seg == 0 ? true : false;
  //   bool isOpd2_16Bit = r->opt_opd2_seg == 0 ? true : false;
  r->tens_lookup = r->tsk_eu_typ == 12 ? 1 : 0;
  r->opd_typ = isRes_16Bit ? 1 : 0;
}
void TiuReg::parse_cv182x_tiu_reg(tiu_reg_t *r, const uint32_t *p) {
  r->cmd_en = p[0] & 1;
  r->cmd_end = (p[0] >> 1) & 1;
  r->cmd_id_en = (p[0] >> 2) & 1;
  r->cmd_keep = (p[0] >> 3) & 1;
  r->cmd_intr_en = (p[0] >> 4) & 1;
  r->tsk_typ = (p[0] >> 5) & ((1u << 4) - 1);
  r->tsk_eu_typ = (p[0] >> 9) & ((1u << 5) - 1);
  r->tsk_opd_num = (p[0] >> 14) & ((1u << 2) - 1);
  r->opt_res_shift = (p[0] >> 16) & ((1u << 6) - 1);
  r->opt_left_shift = (p[0] >> 22) & ((1u << 5) - 1);
  r->opt_shift_typ = (p[0] >> 27) & 1;
  r->opt_rshift_typ = (p[0] >> 28) & 1;
  r->opd_typ = (p[0] >> 30) & 1;
  r->opt_chl_quan = (p[0] >> 31) & 1;
  r->cmd_id_tpu = p[1] & ((1u << 16) - 1);
  r->cmd_id_gdma = (p[1] >> 16) & ((1u << 16) - 1);
  r->quan_m = p[2];
  r->opt_res0_sign = p[3] & 1;
  r->opt_opd0_sign = (p[3] >> 1) & 1;
  r->opt_opd1_sign = (p[3] >> 2) & 1;
  r->opt_opd2_sign = (p[3] >> 3) & 1;
  r->opt_res0_seg = (p[3] >> 4) & ((1u << 2) - 1);
  r->opt_opd0_seg = (p[3] >> 6) & ((1u << 2) - 1);
  r->opt_opd1_seg = (p[3] >> 8) & ((1u << 2) - 1);
  r->opt_opd2_seg = (p[3] >> 10) & 1;
  r->ps32_md = (p[3] >> 11) & ((1u << 2) - 1);
  r->double_conv = (p[3] >> 13) & 1;
  r->opt_left_tran = (p[3] >> 14) & 1;
  r->fp_round_typ = (p[3] >> 15) & 1;
  r->opt_relu_typ = (p[3] >> 16) & ((1u << 2) - 1);
  r->opt_relu_value = (p[3] >> 18) & ((1u << 8) - 1);
  r->cmd_pre_exe_typ = (p[3] >> 26) & 1;
  r->opt_res_add = (p[3] >> 27) & 1;
  r->rsvd0 = (p[3] >> 28) & ((1u << 4) - 1);
  r->conv_opd0_x_ins0 = p[4] & ((1u << 4) - 1);
  r->conv_opd0_y_ins0 = (p[4] >> 4) & ((1u << 4) - 1);
  r->conv_opd0_x_ins0_last = (p[4] >> 8) & ((1u << 4) - 1);
  r->conv_opd0_y_ins0_last = (p[4] >> 12) & ((1u << 4) - 1);
  r->conv_opd1_x_ins0 = (p[4] >> 16) & ((1u << 4) - 1);
  r->conv_opd1_y_ins0 = (p[4] >> 20) & ((1u << 4) - 1);
  r->opd0_ins_val = p[5] & ((1u << 16) - 1);
  r->conv_opd0_up_pad = (p[5] >> 16) & ((1u << 4) - 1);
  r->conv_opd0_dn_pad = (p[5] >> 20) & ((1u << 4) - 1);
  r->conv_opd0_lf_pad = (p[5] >> 24) & ((1u << 4) - 1);
  r->conv_opd0_rt_pad = (p[5] >> 28) & ((1u << 4) - 1);
  r->res0_n = p[6] & ((1u << 12) - 1);
  r->res0_c = (p[6] >> 12) & ((1u << 12) - 1);
  r->res0_h = (p[6] >> 24) & ((1u << 8) - 1);
  r->res0_h |= (uint64_t)(p[7] & ((1u << 4) - 1)) << 8;
  r->res0_w = (p[7] >> 4) & ((1u << 12) - 1);
  r->conv_op_x_str = (p[7] >> 16) & ((1u << 5) - 1);
  r->conv_op_y_str = (p[7] >> 21) & ((1u << 5) - 1);
  r->cmd_pre_exe = (p[7] >> 26) & ((1u << 2) - 1);
  r->rsvd1 = (p[7] >> 28) & ((1u << 4) - 1);
  r->res0_addr = p[8] & ((1u << 24) - 1);
  r->opd0_addr = (p[8] >> 24) & ((1u << 8) - 1);
  r->opd0_addr |= (uint64_t)(p[9] & ((1u << 16) - 1)) << 8;
  r->opd1_addr = (p[9] >> 16) & ((1u << 16) - 1);
  r->opd2_addr = p[10] & ((1u << 16) - 1);
  r->opt_opd0_const = (p[10] >> 16) & 1;
  r->opt_opd1_const = (p[10] >> 17) & 1;
  r->opt_opd2_const = (p[10] >> 18) & 1;
  r->short_nchwstr_same = (p[10] >> 19) & 1;
  r->short_res0_str = (p[10] >> 20) & ((1u << 2) - 1);
  r->short_opd0_str = (p[10] >> 22) & ((1u << 2) - 1);
  r->short_opd1_str = (p[10] >> 24) & ((1u << 2) - 1);
  r->short_opd2_str = (p[10] >> 26) & ((1u << 2) - 1);
  r->opd0_n = p[11] & ((1u << 12) - 1);
  r->opd0_c = (p[11] >> 12) & ((1u << 12) - 1);
  r->rsvd2 = (p[11] >> 28) & ((1u << 4) - 1);
  r->opd0_h = p[12] & ((1u << 12) - 1);
  r->opd0_w = (p[12] >> 12) & ((1u << 12) - 1);
  r->opd1_n = (p[12] >> 24) & ((1u << 8) - 1);
  r->opd1_n |= (uint64_t)(p[13] & ((1u << 4) - 1)) << 8;
  r->opd1_c = (p[13] >> 4) & ((1u << 12) - 1);
  r->opd1_h = (p[13] >> 16) & ((1u << 12) - 1);
  r->opd1_w = (p[13] >> 28) & ((1u << 4) - 1);
  r->opd1_w |= (uint64_t)(p[14] & ((1u << 8) - 1)) << 4;
  r->opd2_n = (p[14] >> 8) & ((1u << 12) - 1);
  r->opd2_c = (p[14] >> 20) & ((1u << 12) - 1);
  r->opd2_h = p[15] & ((1u << 12) - 1);
  r->opd2_w = (p[15] >> 12) & ((1u << 12) - 1);
  r->rsvd3 = (p[15] >> 28) & ((1u << 4) - 1);
  r->layer_info = p[16] & ((1u << 16) - 1);
  r->res0_n_str = (p[16] >> 16) & ((1u << 16) - 1);
  r->res0_c_str = p[17] & ((1u << 16) - 1);
  r->res0_h_str = (p[17] >> 16) & ((1u << 16) - 1);
  r->res0_w_str = p[18] & ((1u << 16) - 1);
  r->res0_b_str = (p[18] >> 16) & ((1u << 16) - 1);
  r->opd0_n_str = p[19] & ((1u << 16) - 1);
  r->rsvd4 = (p[19] >> 28) & ((1u << 4) - 1);
  r->opd0_c_str = p[20] & ((1u << 16) - 1);
  r->opd0_h_str = (p[20] >> 16) & ((1u << 16) - 1);
  r->opd0_w_str = p[21] & ((1u << 16) - 1);
  r->opd0_b_str = (p[21] >> 16) & ((1u << 16) - 1);
  r->opd1_n_str = p[22] & ((1u << 16) - 1);
  r->opd1_c_str = (p[22] >> 16) & ((1u << 16) - 1);
  r->opd1_h_str = p[23] & ((1u << 16) - 1);
  r->rsvd5 = (p[23] >> 28) & ((1u << 4) - 1);
  r->opd1_w_str = p[24] & ((1u << 16) - 1);
  r->opd1_b_str = (p[24] >> 16) & ((1u << 16) - 1);
  r->opd2_n_str = p[25] & ((1u << 16) - 1);
  r->opd2_c_str = (p[25] >> 16) & ((1u << 16) - 1);
  r->opd2_h_str = p[26] & ((1u << 16) - 1);
  r->opd2_w_str = (p[26] >> 16) & ((1u << 16) - 1);
  r->opd2_b_str = p[27] & ((1u << 16) - 1);
  r->rsvd6 = (p[27] >> 28) & ((1u << 4) - 1);
}

void TiuReg::parse_cv183x_tiu_reg(tiu_reg_t *r, const uint32_t *p) {
  r->cmd_en = p[0] & 1;
  r->cmd_end = (p[0] >> 1) & 1;
  r->cmd_id_en = (p[0] >> 2) & 1;
  r->cmd_id_tpu = (p[0] >> 3) & ((1u << 16) - 1);
  r->cmd_id_gdma = (p[0] >> 19) & ((1u << 13) - 1);
  r->cmd_id_gdma |= (uint64_t)(p[1] & ((1u << 3) - 1)) << 13;
  r->cmd_keep = (p[1] >> 3) & 1;
  r->cmd_intr_en = (p[1] >> 4) & 1;
  r->tsk_typ = (p[1] >> 5) & ((1u << 4) - 1);
  r->tsk_eu_typ = (p[1] >> 9) & ((1u << 8) - 1);
  r->tsk_opd_num = (p[1] >> 17) & ((1u << 2) - 1);
  r->opt_right_shift = (p[1] >> 19) & ((1u << 5) - 1);
  r->opt_left_shift = (p[1] >> 24) & ((1u << 5) - 1);
  r->opt_shift_typ = (p[1] >> 29) & 1;
  r->opt_rshift_typ = (p[1] >> 30) & 1;
  r->opt_res_add = (p[1] >> 31) & 1;
  r->opt_relu = p[2] & 1;
  r->opt_left_tran = (p[2] >> 1) & 1;
  r->opt_chl_quan = (p[2] >> 2) & 1;
  r->tens_mdsum = (p[2] >> 3) & 1;
  r->tens_lookup = (p[2] >> 4) & 1;
  r->opt_res0_sign = (p[2] >> 5) & 1;
  r->opt_opd0_sign = (p[2] >> 6) & 1;
  r->opt_opd1_sign = (p[2] >> 7) & 1;
  r->opt_opd2_sign = (p[2] >> 8) & 1;
  r->opt_res0_int8 = (p[2] >> 9) & 1;
  r->opt_opd0_int8 = (p[2] >> 10) & 1;
  r->opt_opd1_int8 = (p[2] >> 11) & 1;
  r->opt_opd2_int8 = (p[2] >> 12) & 1;
  r->opt_opd0_const = (p[2] >> 13) & 1;
  r->opt_opd1_const = (p[2] >> 14) & 1;
  r->opt_opd2_const = (p[2] >> 15) & 1;
  r->short_nchwstr_same = (p[2] >> 16) & 1;
  r->short_res0_str = (p[2] >> 17) & ((1u << 2) - 1);
  r->short_opd0_str = (p[2] >> 19) & ((1u << 2) - 1);
  r->short_opd1_str = (p[2] >> 21) & ((1u << 2) - 1);
  r->short_opd2_str = (p[2] >> 23) & ((1u << 2) - 1);
  r->conv_opd0_x_ins0 = (p[2] >> 25) & ((1u << 4) - 1);
  r->conv_opd0_y_ins0 = (p[2] >> 29) & ((1u << 3) - 1);
  r->conv_opd0_y_ins0 |= (uint64_t)(p[3] & 1) << 3;
  r->conv_opd0_x_ins0_last = (p[3] >> 1) & ((1u << 4) - 1);
  r->conv_opd0_y_ins0_last = (p[3] >> 5) & ((1u << 4) - 1);
  r->conv_opd1_x_ins0 = (p[3] >> 9) & ((1u << 4) - 1);
  r->conv_opd1_y_ins0 = (p[3] >> 13) & ((1u << 4) - 1);
  r->opd0_ins_val = (p[3] >> 17) & ((1u << 8) - 1);
  r->ps32_md = (p[3] >> 25) & ((1u << 2) - 1);
  r->double_conv = (p[3] >> 27) & 1;
  r->rsvd0 = (p[3] >> 28) & ((1u << 4) - 1);
  r->res0_n = p[4] & ((1u << 12) - 1);
  r->res0_c = (p[4] >> 12) & ((1u << 12) - 1);
  r->res0_h = (p[4] >> 24) & ((1u << 8) - 1);
  r->res0_h |= (uint64_t)(p[5] & ((1u << 4) - 1)) << 8;
  r->res0_w = (p[5] >> 4) & ((1u << 12) - 1);
  r->res0_addr = (p[5] >> 16) & ((1u << 16) - 1);
  r->res0_addr |= (uint64_t)(p[6] & ((1u << 8) - 1)) << 16;
  r->opd0_addr = (p[6] >> 8) & ((1u << 24) - 1);
  r->opd1_addr = p[7] & ((1u << 16) - 1);
  r->rsvd1 = (p[7] >> 16) & ((1u << 16) - 1);
  r->opd2_addr = p[8] & ((1u << 16) - 1);
  r->opd0_c = (p[8] >> 16) & ((1u << 12) - 1);
  r->opd0_h = (p[8] >> 28) & ((1u << 4) - 1);
  r->opd0_h |= (uint64_t)(p[9] & ((1u << 8) - 1)) << 4;
  r->opd0_w = (p[9] >> 8) & ((1u << 12) - 1);
  r->opd1_h = (p[9] >> 20) & ((1u << 12) - 1);
  r->opd1_w = p[10] & ((1u << 12) - 1);
  r->conv_opd0_up_pad = (p[10] >> 12) & ((1u << 4) - 1);
  r->conv_opd0_dn_pad = (p[10] >> 16) & ((1u << 4) - 1);
  r->conv_opd0_lf_pad = (p[10] >> 20) & ((1u << 4) - 1);
  r->conv_opd0_rt_pad = (p[10] >> 24) & ((1u << 4) - 1);
  r->conv_op_x_str = (p[10] >> 28) & ((1u << 4) - 1);
  r->conv_op_y_str = p[11] & ((1u << 4) - 1);
  r->opd0_ins_fp = (p[11] >> 4) & ((1u << 16) - 1);
  r->rsvd2 = (p[11] >> 20) & ((1u << 12) - 1);
  r->opd0_n = p[12] & ((1u << 12) - 1);
  r->opd1_n = (p[12] >> 12) & ((1u << 12) - 1);
  r->opd1_c = (p[12] >> 24) & ((1u << 8) - 1);
  r->opd1_c |= (uint64_t)(p[13] & ((1u << 4) - 1)) << 8;
  r->opd2_n = (p[13] >> 4) & ((1u << 12) - 1);
  r->opd2_c = (p[13] >> 16) & ((1u << 12) - 1);
  r->opd2_h = (p[13] >> 28) & ((1u << 4) - 1);
  r->opd2_h |= (uint64_t)(p[14] & ((1u << 8) - 1)) << 4;
  r->opd2_w = (p[14] >> 8) & ((1u << 12) - 1);
  r->quan_m = (p[14] >> 20) & ((1u << 12) - 1);
  r->quan_m |= (uint64_t)(p[15] & ((1u << 20) - 1)) << 12;
  r->opd_typ = (p[15] >> 20) & 1;
  r->fp_round_typ = (p[15] >> 21) & ((1u << 3) - 1);
  r->rsvd7 = (p[15] >> 24) & ((1u << 4) - 1);
  r->rsvd3 = (p[15] >> 28) & ((1u << 4) - 1);
  r->res0_n_str = p[16] & ((1u << 16) - 1);
  r->res0_c_str = (p[16] >> 16) & ((1u << 16) - 1);
  r->res0_h_str = p[17] & ((1u << 16) - 1);
  r->res0_w_str = (p[17] >> 16) & ((1u << 16) - 1);
  r->res0_b_str = p[18] & ((1u << 16) - 1);
  r->opd0_n_str = (p[18] >> 16) & ((1u << 16) - 1);
  r->opd0_c_str = p[19] & ((1u << 16) - 1);
  r->rsvd4 = (p[19] >> 16) & ((1u << 16) - 1);
  r->opd0_h_str = p[20] & ((1u << 16) - 1);
  r->opd0_w_str = (p[20] >> 16) & ((1u << 16) - 1);
  r->opd0_b_str = p[21] & ((1u << 16) - 1);
  r->opd1_n_str = (p[21] >> 16) & ((1u << 16) - 1);
  r->opd1_c_str = p[22] & ((1u << 16) - 1);
  r->opd1_h_str = (p[22] >> 16) & ((1u << 16) - 1);
  r->opd1_w_str = p[23] & ((1u << 16) - 1);
  r->rsvd5 = (p[23] >> 16) & ((1u << 16) - 1);
  r->opd1_b_str = p[24] & ((1u << 16) - 1);
  r->opd2_n_str = (p[24] >> 16) & ((1u << 16) - 1);
  r->opd2_c_str = p[25] & ((1u << 16) - 1);
  r->opd2_h_str = (p[25] >> 16) & ((1u << 16) - 1);
  r->opd2_w_str = p[26] & ((1u << 16) - 1);
  r->opd2_b_str = (p[26] >> 16) & ((1u << 16) - 1);
  r->layer_info = p[27] & ((1u << 28) - 1);
  r->rsvd6 = (p[27] >> 28) & ((1u << 4) - 1);
  r->cmd_pre_exe = 0;
  r->opt_res0_seg = 0;
  r->opt_opd0_seg = 0;
  r->opt_opd1_seg = 0;
  r->opt_opd2_seg = 0;
  r->opt_relu_typ = 0;
  r->opt_relu_value = 0;
  r->opt_res_shift = 0;
  r->cmd_pre_exe_typ = 0;
}

uint64_t TdmaReg::calCycle(tdma_reg_t task, uint64_t dram_frequency,
                           uint64_t sram_frequency) {
  uint64_t dram_count = 0;
  uint64_t sram_count = 0;

  if (task.trans_dir == 0 || task.trans_dir == 2) {
    calLoad(task, dram_count, sram_count);
  } else if (task.trans_dir == 1) {
    calStore(task, dram_count, sram_count);
  } else if (task.trans_dir == 3) {
    dram_count = calMove(task);
  } else {
    assert(0 && "unsupported tdma mode");
  }

  uint64_t dram_byte_width = 32 / 8;
  uint64_t dram_bw = dram_frequency * dram_byte_width;
  uint64_t sram_byte_width = LOCAL_MEM_WIDTH;
  uint64_t sram_bw = sram_frequency * sram_byte_width;

  uint64_t dram_time = dram_count * 1000 / dram_bw;
  uint64_t sram_time = sram_count * 1000 / sram_bw;

  return dram_time > sram_time ? dram_time : sram_time;
}

void TdmaReg::parse_tdma_reg(tdma_reg_t *r, const uint32_t *p,
                             unsigned char magicNum) {
  if (magicNum == 0xA5)
    parse_cv183x_tdma_reg(r, p);
  else if (magicNum == 0xA6) {
    parse_cv182x_tdma_reg(r, p);
  } else if (magicNum == 0xA7) {
    parse_cv182x_tdma_reg(r, p);
  } else if (magicNum == 0xA8) {
    parse_cv182x_tdma_reg(r, p);
  } else {
    printf("invalid magic number %x\n", magicNum);
    assert(0);
  }
}

void TdmaReg::calLoad(tdma_reg_t task, uint64_t &dram_count,
                      uint64_t &sram_count) {
  if (task.sys_dtype == 1) { // matrix
    // Force src_n to 1
    task.src_h = 1;
    task.src_h_stride =
        (task.src_c_stride_high << 16) | (task.src_c_stride_low);
    task.src_n = 1;
  }
  bool isMatrix = (task.sys_dtype == 1);
  bool isTranspose =
      isMatrix ? ((task.spec_func == 1) && (task.transpose_md == 0)) : false;
  bool isHwcMode = (task.spec_func == 1) &&
                   ((task.transpose_md == 1) || (task.transpose_md == 2));
  if (isHwcMode) {
    int realC = task.src_c;
    int realW = task.src_w;
    task.src_w = realC;
    task.src_h_stride = realC;
    task.src_c = realW;
    task.src_c_stride_low = task.src_h_stride * task.src_h;
  }
  int c_stride = (task.src_c_stride_high << 16) | (task.src_c_stride_low);
  bool isHContinuous =
      isTranspose ? false
                  : (task.src_h_stride - task.src_w <= DATA_MAX_DISTANCE);
  bool isCContinuous = isHContinuous && (c_stride - task.src_w * task.src_h <=
                                         DATA_MAX_DISTANCE);
  bool isSrcBf16 = (task.src_fmt == 2);
  int srcDataSize = isSrcBf16 ? 2 : 1;
  int h_last_valid =
      (task.src_h_stride) * (task.src_h - 1) + task.src_w * srcDataSize;
  int c_last_valid = c_stride * (task.src_c - 1) + h_last_valid;
  int generalCopySize = task.src_n_stride;
  bool isGeneralMove = task.trans_fmt;

  if (isGeneralMove) {
    uint64_t baseAddr = get_src_address(task);
    dram_count = get_tdma_cycle(task, baseAddr, generalCopySize, false);
  } else {
    if (isCContinuous) {
      uint64_t baseAddr = get_src_address(task);
      for (int n = 0; n < (int)task.src_n; n++) {
        uint64_t addr = baseAddr + (task.src_n_stride * n);
        dram_count = get_tdma_cycle(task, addr, c_last_valid, false);
      }
    } else if (isHContinuous) {
      uint64_t baseAddr = get_src_address(task);
      for (int n = 0; n < (int)task.src_n; n++) {
        for (int c = 0; c < (int)task.src_c; c++) {
          uint64_t addr = baseAddr + task.src_n_stride * n + c_stride * c;
          dram_count = get_tdma_cycle(task, addr, h_last_valid, false);
        }
      }
    } else {
      if (isTranspose) {
        const int localSramRowSize = 8;
        const int localSramColSize = 8 * 16;
        uint64_t baseAddr = get_src_address(task);
        for (int c = 0; c < (int)task.src_c; c += localSramRowSize) {
          for (int w = 0; w < (int)task.src_w; w += localSramColSize) {
            int loopRowSize = (task.src_c - c >= localSramRowSize)
                                  ? localSramRowSize
                                  : task.src_c - c;
            for (int k = 0; k < loopRowSize; k++) {
              uint64_t addr = baseAddr + c_stride * (c + k) + w;
              int size = (task.src_w - w >= localSramColSize) ? localSramColSize
                                                              : task.src_w - w;
              dram_count = get_tdma_cycle(task, addr, size, false);
            }
          }
        }
      } else {
        uint64_t baseAddr = get_src_address(task);
        for (int n = 0; n < (int)task.src_n; n++) {
          for (int c = 0; c < (int)task.src_c; c++) {
            for (int h = 0; h < (int)task.src_h; h++) {
              uint64_t addr = baseAddr + task.src_n_stride * n + c_stride * c +
                              task.src_h_stride * h;
              uint64_t data_size = task.src_w * srcDataSize;
              dram_count = get_tdma_cycle(task, addr, data_size, false);
            }
          }
        }
      }
    }
  }
  sram_count = calSramCycle(task);
}

// copy from TdmaStorer.cc
void TdmaReg::calStore(tdma_reg_t &task, uint64_t &dram_count,
                       uint64_t &sram_count) {
  int dst_n = task.src_n;
  bool isNcTranspose = (task.transpose_md == 0);
  bool isSpecialFunction = (task.spec_func == 1);
  if (task.sys_dtype == 1) {
    // Force src_n to 1
    task.dst_h = 1;
    task.dst_h_stride =
        (task.dst_c_stride_high << 16) | (task.dst_c_stride_low);
    dst_n = 1;
  }
  if (isSpecialFunction && isNcTranspose) {
    dst_n = task.src_c;
  }
  // cout << tdmaDesTskTypeName[task.funcName] << endl;
  int c_stride = (task.dst_c_stride_high << 16) | (task.dst_c_stride_low);
  bool isDstBf16 = (task.dst_fmt == 2);
  int dataSize = isDstBf16 ? 2 : 1;
  bool isCwTranspose = (task.transpose_md == 3);
  bool isHContinuous = isCwTranspose
                           ? false
                           : (task.dst_h_stride - task.dst_w * dataSize <=
                              STORE_DATA_MAX_DISTANCE);
  bool isCContinuous = (c_stride - task.dst_w * task.dst_h * dataSize <=
                        STORE_DATA_MAX_DISTANCE) &&
                       isHContinuous;
  int h_last_valid =
      (task.dst_h_stride) * (task.dst_h - 1) + task.dst_w * dataSize;
  int c_last_valid = c_stride * (task.dst_c - 1) + h_last_valid;
  int generalCopySize = task.src_n_stride;
  bool isGeneralMove = task.trans_fmt;

  if (isGeneralMove) {
    uint64_t baseAddr = get_dst_address(task);
    dram_count = get_tdma_cycle(task, baseAddr, generalCopySize, true);
  } else {
    if (isCContinuous) {
      for (int i = 0; i < dst_n; i++) {
        uint64_t baseAddr = get_dst_address(task) + task.dst_n_stride * i;
        dram_count = get_tdma_cycle(task, baseAddr, c_last_valid, true);
      }
    } else if (isHContinuous) {
      for (int i = 0; i < (int)dst_n; i++) {
        for (int j = 0; j < (int)task.dst_c; j++) {
          uint64_t baseAddr =
              get_dst_address(task) + task.dst_n_stride * i + c_stride * j;
          dram_count = get_tdma_cycle(task, baseAddr, h_last_valid, true);
        }
      }
    } else {
      for (int i = 0; i < (int)dst_n; i++) {
        for (int j = 0; j < (int)task.dst_c; j++) {
          for (int k = 0; k < (int)task.dst_h; k++) {
            uint64_t baseAddr = get_dst_address(task) + task.dst_n_stride * i +
                                c_stride * j + task.dst_h_stride * k;
            uint64_t data_size = task.dst_w * dataSize;
            dram_count = get_tdma_cycle(task, baseAddr, data_size, true);
          }
        }
      }
    }
  }
  sram_count = calSramCycle(task);
}

uint64_t TdmaReg::calMove(tdma_reg_t &task) {
  // Todo : simulate bank conflict
  bool isSrcBf16 = (task.src_fmt == 2);
  bool isDstBf16 = (task.dst_fmt == 2);
  int srcAtomicSize = isSrcBf16 ? 2 : 1;
  int dstAtomicSize = isDstBf16 ? 2 : 1;
  bool isLoadHContinuous = (task.src_h_stride - task.src_w == 0);
  int load_h_last_valid =
      (task.src_h_stride) * (task.src_h - 1) + task.src_w * srcAtomicSize;
  int loadCycleTime =
      (isLoadHContinuous)
          ? task.src_n * task.src_c *
                align_up(load_h_last_valid, LOCAL_MEM_WIDTH)
          : task.src_n * task.src_c * task.src_h *
                align_up(task.src_w * srcAtomicSize,
                         static_cast<uint32_t>(LOCAL_MEM_WIDTH));

  bool isStoreHContinuous = (task.dst_h_stride - task.dst_w == 0);
  int store_h_last_valid =
      (task.dst_h_stride) * (task.dst_h - 1) + task.dst_w * dstAtomicSize;
  int storeCycleTime =
      (isStoreHContinuous)
          ? task.src_n * task.dst_c *
                align_up(store_h_last_valid, LOCAL_MEM_WIDTH)
          : task.src_n * task.dst_c * task.dst_h *
                align_up(task.dst_w * dstAtomicSize,
                         static_cast<uint32_t>(LOCAL_MEM_WIDTH));
  return loadCycleTime > storeCycleTime ? loadCycleTime : storeCycleTime;
}

uint64_t TdmaReg::get_src_address(tdma_reg_t &r) {
  uint64_t addr =
      (((uint64_t)r.src_base_addr_high << SRC_BASE_ADDR_HIGH_SHIFT) |
       r.src_base_addr_low);
  return addr;
}

uint64_t TdmaReg::get_dst_address(tdma_reg_t &r) {
  uint64_t addr =
      (((uint64_t)r.dst_base_addr_high << SRC_BASE_ADDR_HIGH_SHIFT) |
       r.dst_base_addr_low);
  return addr;
}

uint64_t TdmaReg::get_tdma_cycle(tdma_reg_t &task, uint64_t baseAddr,
                                 uint64_t data_size, bool isStore) {
  uint64_t dram_byte_count = 0;
  bool isCwTranspose = (task.transpose_md == 3);
  int max_burst_length = MAX_BURST_LENGTH;
  if (isCwTranspose && isStore)
    max_burst_length = SPECIAL_FUNCTION_BURST_LENGTH;
  int max_burst_size = max_burst_length * AXI_BUS_WIDTH;

  bool isCross4KBoundary = ((baseAddr & FOUR_KB_MASK) !=
                            ((((baseAddr + data_size))) & FOUR_KB_MASK));
  if (isCross4KBoundary) {
    int nearest4KBBoundary = align_up(baseAddr, static_cast<uint64_t>(FOUR_KB));
    if (baseAddr != (uint64_t)nearest4KBBoundary) {
      int headOffsetTo4K = ((nearest4KBBoundary)) - ((baseAddr));
      int headBurstNumber = headOffsetTo4K / (max_burst_size);
      int remainSizeForBurst =
          align_up(headOffsetTo4K % max_burst_size, AXI_BUS_WIDTH);
      for (int packetNum = 0; packetNum < headBurstNumber; packetNum++) {
        uint64_t tempBasedAddr = baseAddr + packetNum * max_burst_size;
        tempBasedAddr = (packetNum == 0)
                            ? tempBasedAddr
                            : align_down(tempBasedAddr, AXI_BUS_WIDTH);
        uint64_t tempByteCnt = calByteCnt(tempBasedAddr, max_burst_size);
        dram_byte_count += tempByteCnt;
      }
      if (remainSizeForBurst > 0) {
        uint64_t tempBasedAddr = baseAddr + headBurstNumber * max_burst_size;
        tempBasedAddr = (headBurstNumber == 0)
                            ? tempBasedAddr
                            : align_down(tempBasedAddr, AXI_BUS_WIDTH);
        uint64_t tempByteCnt = calByteCnt(tempBasedAddr, remainSizeForBurst);
        dram_byte_count += tempByteCnt;
      }
    }
    int tailOffsetTo4K = ((baseAddr + data_size)) - ((nearest4KBBoundary));
    int tailBurstNumber = tailOffsetTo4K / (max_burst_size);
    int tailRemainSizeForBurst =
        align_up(tailOffsetTo4K % (max_burst_size), AXI_BUS_WIDTH);
    for (int packetNum = 0; packetNum < tailBurstNumber; packetNum++) {
      uint64_t tempBasedAddr = nearest4KBBoundary + packetNum * max_burst_size;
      tempBasedAddr = (packetNum == 0)
                          ? tempBasedAddr
                          : align_down(tempBasedAddr, AXI_BUS_WIDTH);
      uint64_t tempByteCnt = calByteCnt(tempBasedAddr, max_burst_size);
      dram_byte_count += tempByteCnt;
    }
    if (tailRemainSizeForBurst > 0) {
      uint64_t tempBasedAddr =
          nearest4KBBoundary + tailBurstNumber * max_burst_size;
      tempBasedAddr = align_down(tempBasedAddr, AXI_BUS_WIDTH);
      uint64_t tempByteCnt = calByteCnt(tempBasedAddr, tailRemainSizeForBurst);
      dram_byte_count += tempByteCnt;
    }
  } else {
    int addressRangePeriod = baseAddr + data_size - align_down(baseAddr, 16);
    int prevMaxBurstNumber = addressRangePeriod / (max_burst_size);
    int prevRemainSizeForBurst =
        align_up(addressRangePeriod % (max_burst_size), AXI_BUS_WIDTH);
    for (int packetNum = 0; packetNum < prevMaxBurstNumber; packetNum++) {
      uint64_t tempBasedAddr = baseAddr + packetNum * max_burst_size;
      tempBasedAddr = (packetNum == 0)
                          ? tempBasedAddr
                          : align_down(tempBasedAddr, AXI_BUS_WIDTH);
      uint64_t tempByteCnt = calByteCnt(tempBasedAddr, max_burst_size);
      dram_byte_count += tempByteCnt;
    }
    if (prevRemainSizeForBurst) {
      uint64_t tempBasedAddr = baseAddr + prevMaxBurstNumber * max_burst_size;
      tempBasedAddr = (prevMaxBurstNumber == 0)
                          ? tempBasedAddr
                          : align_down(tempBasedAddr, AXI_BUS_WIDTH);
      uint64_t tempByteCnt = calByteCnt(tempBasedAddr, prevRemainSizeForBurst);
      dram_byte_count += tempByteCnt;
    }
  }

  return dram_byte_count;
}
uint64_t TdmaReg::calByteCnt(uint64_t baseAddr, uint64_t size) {
  uint64_t tempBaseAddrAlign16Byte = align_down(baseAddr, AXI_BUS_WIDTH);
  uint64_t tempEndAddrAlign16Byte = tempBaseAddrAlign16Byte + size;
  uint64_t nearestTempBasedAddr64BBoundary =
      align_up(tempBaseAddrAlign16Byte, static_cast<uint64_t>(BYTE64));
  uint64_t nearestTempEndAddr64BBoundary =
      align_up(tempEndAddrAlign16Byte, static_cast<uint64_t>(BYTE64));
  uint64_t tempByteCnt =
      !(tempBaseAddrAlign16Byte == nearestTempBasedAddr64BBoundary) +
      (nearestTempEndAddr64BBoundary - nearestTempBasedAddr64BBoundary) /
          BYTE64;
  tempByteCnt = tempByteCnt * BYTE64;
  return tempByteCnt;
}

uint64_t TdmaReg::calSramCycle(tdma_reg_t &task) {
  bool isDstBf16 = (task.dst_fmt == 2);
  uint64_t dataSize = isDstBf16 ? 2 : 1;

  bool isStoreHContinuous = (task.dst_h_stride - task.dst_w == 0);
  int store_h_last_valid =
      (task.dst_h_stride) * (task.dst_h - 1) + task.dst_w * dataSize;
  uint64_t storeCycleTime =
      (isStoreHContinuous)
          ? task.src_n * task.dst_c *
                ceiling_func(store_h_last_valid, LOCAL_MEM_WIDTH)
          : task.src_n * task.dst_c * task.dst_h *
                ceiling_func(task.dst_w * dataSize,
                             static_cast<uint64_t>(LOCAL_MEM_WIDTH));
  return storeCycleTime;
}

void TdmaReg::parse_cv182x_tdma_reg(tdma_reg_t *r, const uint32_t *p) {
  r->vld = p[0] & 1;
  r->compress_en = (p[0] >> 1) & 1;
  r->eod = (p[0] >> 2) & 1;
  r->intp_en = (p[0] >> 3) & 1;
  r->bar_en = (p[0] >> 4) & 1;
  r->check_bf16_value = (p[0] >> 5) & 1;
  r->trans_dir = (p[0] >> 6) & ((1u << 2) - 1);
  r->rsv00 = (p[0] >> 8) & ((1u << 2) - 1);
  r->trans_fmt = (p[0] >> 10) & 1;
  r->transpose_md = (p[0] >> 11) & ((1u << 2) - 1);
  r->rsv01 = (p[0] >> 13) & 1;
  r->intra_cmd_paral = (p[0] >> 14) & 1;
  r->outstanding_en = (p[0] >> 15) & 1;
  r->cmd_id = (p[0] >> 16) & ((1u << 16) - 1);
  r->spec_func = p[1] & ((1u << 3) - 1);
  r->dst_fmt = (p[1] >> 3) & ((1u << 2) - 1);
  r->src_fmt = (p[1] >> 5) & ((1u << 2) - 1);
  r->cmprs_fmt = (p[1] >> 7) & 1;
  r->sys_dtype = (p[1] >> 8) & 1;
  r->rsv2_1 = (p[1] >> 9) & ((1u << 4) - 1);
  r->int8_sign = (p[1] >> 13) & 1;
  r->compress_zero_guard = (p[1] >> 14) & 1;
  r->int8_rnd_mode = (p[1] >> 15) & 1;
  r->wait_id_tpu = (p[1] >> 16) & ((1u << 16) - 1);
  r->wait_id_other_tdma = p[2] & ((1u << 16) - 1);
  r->wait_id_sdma = (p[2] >> 16) & ((1u << 16) - 1);
  r->const_val = p[3] & ((1u << 16) - 1);
  r->src_base_reg_sel = (p[3] >> 16) & ((1u << 3) - 1);
  r->mv_lut_idx = (p[3] >> 19) & 1;
  r->dst_base_reg_sel = (p[3] >> 20) & ((1u << 3) - 1);
  r->mv_lut_base = (p[3] >> 23) & 1;
  r->rsv4_5 = (p[3] >> 24) & ((1u << 8) - 1);
  r->dst_h_stride = p[4] & ((1u << 16) - 1);
  r->dst_c_stride_low = (p[4] >> 16) & ((1u << 16) - 1);
  r->dst_n_stride = p[5];
  r->src_h_stride = p[6] & ((1u << 16) - 1);
  r->src_c_stride_low = (p[6] >> 16) & ((1u << 16) - 1);
  r->src_n_stride = p[7];
  r->dst_c = p[8] & ((1u << 16) - 1);
  r->src_c = (p[8] >> 16) & ((1u << 16) - 1);
  r->dst_w = p[9] & ((1u << 16) - 1);
  r->dst_h = (p[9] >> 16) & ((1u << 16) - 1);
  r->src_w = p[10] & ((1u << 16) - 1);
  r->src_h = (p[10] >> 16) & ((1u << 16) - 1);
  r->dst_base_addr_low = p[11];
  r->src_base_addr_low = p[12];
  r->src_n = p[13] & ((1u << 16) - 1);
  r->dst_base_addr_high = (p[13] >> 16) & ((1u << 8) - 1);
  r->src_base_addr_high = (p[13] >> 24) & ((1u << 8) - 1);
  r->src_c_stride_high = p[14] & ((1u << 16) - 1);
  r->dst_c_stride_high = (p[14] >> 16) & ((1u << 16) - 1);
  r->compress_bias0 = p[15] & ((1u << 8) - 1);
  r->compress_bias1 = (p[15] >> 8) & ((1u << 8) - 1);
  r->layer_ID = (p[15] >> 16) & ((1u << 16) - 1);
}

void TdmaReg::parse_cv183x_tdma_reg(tdma_reg_t *r, const uint32_t *p) {
  r->vld = p[0] & 1;
  r->compress_en = (p[0] >> 1) & 1;
  r->eod = (p[0] >> 2) & 1;
  r->intp_en = (p[0] >> 3) & 1;
  r->bar_en = (p[0] >> 4) & 1;
  r->check_bf16_value = (p[0] >> 5) & 1;
  r->trans_dir = (p[0] >> 6) & ((1u << 2) - 1);
  r->rsv00 = (p[0] >> 8) & ((1u << 2) - 1);
  r->trans_fmt = (p[0] >> 10) & 1;
  r->transpose_md = (p[0] >> 11) & ((1u << 2) - 1);
  r->rsv01 = (p[0] >> 13) & ((1u << 2) - 1);
  r->outstanding_en = (p[0] >> 15) & 1;
  r->cmd_id = (p[0] >> 16) & ((1u << 16) - 1);
  r->spec_func = p[1] & ((1u << 3) - 1);
  r->dst_fmt = (p[1] >> 3) & ((1u << 2) - 1);
  r->src_fmt = (p[1] >> 5) & ((1u << 2) - 1);
  r->cmprs_fmt = (p[1] >> 7) & 1;
  r->sys_dtype = (p[1] >> 8) & 1;
  r->rsv2_1 = (p[1] >> 9) & ((1u << 4) - 1);
  r->int8_sign = (p[1] >> 13) & 1;
  r->compress_zero_guard = (p[1] >> 14) & 1;
  r->int8_rnd_mode = (p[1] >> 15) & 1;
  r->wait_id_tpu = (p[1] >> 16) & ((1u << 16) - 1);
  r->wait_id_other_tdma = p[2] & ((1u << 16) - 1);
  r->wait_id_sdma = (p[2] >> 16) & ((1u << 16) - 1);
  r->const_val = p[3] & ((1u << 16) - 1);
  r->src_base_reg_sel = (p[3] >> 16) & ((1u << 3) - 1);
  r->mv_lut_idx = (p[3] >> 19) & 1;
  r->dst_base_reg_sel = (p[3] >> 20) & ((1u << 3) - 1);
  r->mv_lut_base = (p[3] >> 23) & 1;
  r->rsv4_5 = (p[3] >> 24) & ((1u << 8) - 1);
  r->dst_h_stride = p[4] & ((1u << 16) - 1);
  r->dst_c_stride_low = (p[4] >> 16) & ((1u << 16) - 1);
  r->dst_n_stride = p[5];
  r->src_h_stride = p[6] & ((1u << 16) - 1);
  r->src_c_stride_low = (p[6] >> 16) & ((1u << 16) - 1);
  r->src_n_stride = p[7];
  r->dst_c = p[8] & ((1u << 16) - 1);
  r->src_c = (p[8] >> 16) & ((1u << 16) - 1);
  r->dst_w = p[9] & ((1u << 16) - 1);
  r->dst_h = (p[9] >> 16) & ((1u << 16) - 1);
  r->src_w = p[10] & ((1u << 16) - 1);
  r->src_h = (p[10] >> 16) & ((1u << 16) - 1);
  r->dst_base_addr_low = p[11];
  r->src_base_addr_low = p[12];
  r->src_n = p[13] & ((1u << 16) - 1);
  r->dst_base_addr_high = (p[13] >> 16) & ((1u << 8) - 1);
  r->src_base_addr_high = (p[13] >> 24) & ((1u << 8) - 1);
  r->src_c_stride_high = p[14] & ((1u << 16) - 1);
  r->dst_c_stride_high = (p[14] >> 16) & ((1u << 16) - 1);
  r->compress_bias0 = p[15] & ((1u << 8) - 1);
  r->compress_bias1 = (p[15] >> 8) & ((1u << 8) - 1);
  r->layer_ID = (p[15] >> 16) & ((1u << 16) - 1);
  r->intra_cmd_paral = 0;
}
} // namespace backend
} // namespace tpu_mlir
