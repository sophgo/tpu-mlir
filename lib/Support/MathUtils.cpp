#include "sophgo/Support/MathUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "float.h"
#include <map>

namespace sophgo {

void get_scale_and_shift(float scale_f, int &scale, int &shift, int bitwidth) {
  float min_err = FLT_MAX;
  int m_limit = (bitwidth == 32) ? INT_MAX : CHAR_MAX;
  for (
      int n = -32; n < 31;
      n++) { //若scale_f大于等于1，这里循环上限要设为31(而不是32)，且越大则需减少越多，暂只考虑scale_f小于等于1的情形
             // wxc 20220119
    int m = (int)std::round(scale_f * std::pow(2, n));
    float err = std::abs(m / std::pow(2, n) - scale_f);
    if (err < min_err && abs(m) < m_limit) {
      min_err = err;
      shift = n;
    }
  }
  scale = (int)std::round(scale_f * std::pow(2, shift));
}

void get_scale_and_shift_positive(float scale_f, int &scale, int &shift,
                                  int bitwidth) {
  float min_err = FLT_MAX;
  int m_limit = (bitwidth == 32) ? INT_MAX : CHAR_MAX;
  for (int n = 0; n < 31; n++) {
    int m = (int)std::round(scale_f * std::pow(2, n));
    float err = std::abs(m / std::pow(2, n) - scale_f);
    if (err < min_err && abs(m) < m_limit) {
      min_err = err;
      shift = n;
    }
  }
  scale = (int)std::round(scale_f * std::pow(2, shift));
}

// this function search positive right shift with max_shift, max_shift set to 8
// for int16 op and shift to 8bit output.
void get_scale_and_shift_positive_maxshift(float scale_f, int &scale,
                                           int &shift, int bitwidth,
                                           int max_shift) {
  float min_err = FLT_MAX;
  int m_limit = (bitwidth == 32) ? INT_MAX : CHAR_MAX;
  for (int n = 0; n < max_shift; n++) {
    int m = (int)std::round(scale_f * std::pow(2, n));
    float err = std::abs(m / std::pow(2, n) - scale_f);
    if (err < min_err && abs(m) < m_limit) {
      min_err = err;
      shift = n;
    }
  }
  scale = (int)std::round(scale_f * std::pow(2, shift));
}

template <typename Dtype> float findMaxabs(const Dtype *pSrcData, int len) {
  float fmax = 0.0;
  float dataTmp;
  for (int i = 0; i < len; i++) {
    dataTmp = (float)pSrcData[i];
    fmax = (fabs(dataTmp) > fmax) ? fabs(dataTmp) : fmax;
  }
  if (fmax == 0.0) {
    ; // LOG(WARNING) << "findMaxabs meet fmax == 0";
  }

  return fmax;
}
template float findMaxabs<float>(const float *pSrcData, int len);
template float findMaxabs<int>(const int *pSrcData, int len);

template <typename Dtype>
void findMinMax(const Dtype *pSrcData, int len, float *minVal, float *maxVal) {
  float fmin = (float)pSrcData[0];
  float fmax = (float)pSrcData[0];
  float dataTmp;
  for (int i = 0; i < len; i++) {
    dataTmp = (float)pSrcData[i];
    fmin = dataTmp < fmin ? dataTmp : fmin;
    fmax = dataTmp > fmax ? dataTmp : fmax;
  }
  *minVal = fmin;
  *maxVal = fmax;
}
template void findMinMax<float>(const float *pSrcData, int len, float *minVal,
                                float *maxVal);
template void findMinMax<int>(const int *pSrcData, int len, float *minVal,
                              float *maxVal);

int calRightShiftNum(float fmax, double thBottom, double thTop, int numBits) {
  double dataTmp = 1.0 * fmax * thBottom / thTop;
  int m = 0;

  if (dataTmp <= 0.0) {
    llvm::errs() << "meet dataTmp <= 0.0, fmax:" << fmax
                 << " thBottom:" << thBottom << " thTop:" << thTop;
    return 0;
  }
  while (dataTmp < ((1 << (numBits - 1)) - 1)) {
    dataTmp = dataTmp * 2;
    m = m + 1;
  }

  m = m > 32 ? 31 : m - 1;
  return m;
}

template <typename T> void func_abs(int n, T *src, T *dst) {
  for (int i = 0; i < n; i++) {
    dst[i] = std::abs(src[i]);
  }
}

template <typename T> void func_log(int n, T *src, T *dst) {
  for (int i = 0; i < n; i++) {
    dst[i] = std::log(src[i]);
  }
}

int calRightShiftNumUseCblas(float fmax, double thBottom, double thTop,
                             int numBits) {
  func_abs(1, &fmax, &fmax);
  double dataTmp = 1.0 * ((1 << (numBits - 1)) - 1) / (fmax * thBottom / thTop);
  int m = 0;

  double log_dem, log_num;
  double data2 = 2.0;

  func_log(1, &dataTmp, &log_dem);
  func_log(1, &data2, &log_num);

  m = floor(log_dem / log_num);
  m = m > 31 ? 31 : m;

  return m;
}

float func_log2(double dataInput) {
  double log_dem, log_num;
  double data2 = 2.0;
  float result;

  func_log(1, &dataInput, &log_dem);
  func_log(1, &data2, &log_num);

  result = log_dem / log_num;

  return result;
}

float quantizeToInt16(const float *pSrc, int16_t *pDst, int len, float scale,
                      int rshift) {
  int16_t qmax = 32767;
  int16_t qmin = -32768;
  int tmp = 0;
  int overflow = 0;
  int half_data = (rshift == 0) ? 0 : 1 << (rshift - 1);

  for (int i = 0; i < len; i++) {
    tmp = round(pSrc[i] * scale);
    tmp = (tmp + half_data) >> rshift;
    pDst[i] = (int16_t)((tmp > qmax) ? qmax : ((tmp < qmin) ? qmin : tmp));
    overflow = (tmp > qmax || tmp < qmin) ? overflow + 1 : overflow;
  }
  float ratio = ((float)overflow) / ((float)len);
  if (ratio > 0 && len > 1) {
    llvm::errs() << "ratio of overflow = " << ratio;
  }
  return ratio;
}

float quantizeToInt15(const float *pSrc, int16_t *pDst, int len, float scale,
                      int rshift) {
  int16_t qmax = 16383;
  int16_t qmin = -16384;
  int tmp = 0;
  int overflow = 0;
  int half_data = (rshift == 0) ? 0 : 1 << (rshift - 1);

  for (int i = 0; i < len; i++) {
    tmp = floor(pSrc[i] * scale + 0.5);
    tmp = (tmp + half_data) >> rshift;
    pDst[i] = (int16_t)((tmp > qmax) ? qmax : ((tmp < qmin) ? qmin : tmp));
    overflow = (tmp > qmax || tmp < qmin) ? overflow + 1 : overflow;
  }
  float ratio = ((float)overflow) / ((float)len);
  if (ratio > 0) {
    llvm::errs() << "ratio of overflow = " << ratio;
  }
  return ratio;
}

void quantizeToInt8(const float *pSrc, int8_t *pDst, int len, float scale) {
  int8_t qmax = 127;
  int8_t qmin = -128;
  int tmp = 0;

  for (int i = 0; i < len; i++) {
    tmp = round(pSrc[i] * scale);
    pDst[i] = (int8_t)((tmp > qmax) ? qmax : ((tmp < qmin) ? qmin : tmp));
  }
}

void pad_tensor(float* input,  float* input_paded1,  float* input_paded2, int n, int ic, int ih, int iw, int pt, int pb, int pl, int pr) {
  float* _input_paded1 = input_paded1;
  float* _input_paded2 = input_paded2;
  int _ih = ih - pt - pb;
  int _iw = iw - pl - pr;

  if (pt > 0 || pb > 0 || pr > 0 || pl > 0) {
    /*int chan_idx = 10;
    int chan_offset = chan_idx*_ih*_iw;
    printf("N:0 C:0 input tensor(%d,%d):\n", _ih, _iw);
    for (int i= 0; i< _ih; i++) {
        for (int j= 0; j< _iw; j++) {
          printf("%d ", (int)(*(input+chan_offset+i*_iw+j)));
        }
        printf("***\n");
    }*/

    for (int i= 0; i< n*ic; i++) {
      _input_paded1 += pt * _iw;
      memcpy(_input_paded1, input + _ih * _iw * i, sizeof(float) * _ih * _iw);
      _input_paded1 += _ih * _iw;
      _input_paded1 += pb * _iw;
    }

    for (int i= 0; i< n*ic*(_ih+pt+pb); i++) {
      _input_paded2 += pl;
      memcpy(_input_paded2, input_paded1 + _iw * i, sizeof(float) * _iw);
      _input_paded2 += _iw;
      _input_paded2 += pr;
    }

    /*chan_offset = chan_idx*ih*iw;
    printf("N:0 C:0 after paded tensor(%d,%d):\n", ih, iw);
    for (int i= 0; i< ih; i++) {
        for (int j= 0; j< iw; j++) {
          printf("%d ", (int)(*(input_paded2+chan_offset+i*iw+j)));
        }
        printf("***\n");
    }*/
  }

#if 0
  if (_pt > 0 || _pb > 0 || _pr > 0 || _pl > 0) {
    int n = src_shape[0];
    int ic = src_shape[1];
    int ih = src_shape[2];
    int iw = src_shape[3];
    for (int j = 0; j < n*ic; j++) {
      // Padding on the low end
      mem_set(_pt * iw, _izp, _input_paded1);
      _input_paded1 += _pt * iw;
      // Copy the original value
      const float *input_data = _input + ih * iw * j;
      mem_copy(ih * iw, input_data, _input_paded1);
      _input_paded1 += ih * iw;
      // Padding on the high end
      mem_set(_pb * iw, _izp, _input_paded1);
      _input_paded1 += _pb * iw;
    }

    for (int j = 0; j < n*ic*(ih+_pt+_pb); j++) {
      // Padding on the low end
      mem_set(_pl, _izp, _input_paded2);
      _input_paded2 += _pl;
      // Copy the original value
      const auto *input_data = _input_paded1 + iw * j;
      ufw_copy(iw, input_data, _input_paded2);
      _input_paded2 += iw;
      // Padding on the high end
      mem_set(_pr, _izp, _input_paded2);
      _input_paded2 += _pr;
    }
  }
#endif
}

} // namespace sophgo
