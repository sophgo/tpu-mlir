//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include <execinfo.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define MULTI_THREAD_KL_CALC
#ifdef MULTI_THREAD_KL_CALC
#include <pthread.h>
#include <time.h>
#endif

extern "C" {

#ifdef MULTI_THREAD_KL_CALC
struct mul_thread_inputs {
  long long *hist;
  float *kl;
  long long count;
  long long i;
  long long N;
  long long BINS;
};
#endif

static inline void print_trace(void) {
  void *array[10];
  size_t i;

  size_t size = backtrace(array, 10);
  char **strings = backtrace_symbols(array, size);

  printf("Obtained %lu stack frames.\n", size);

  for (i = 0; i < size; i++)
    printf("%s\n", strings[i]);

  free(strings);
}

static inline void hang(long long ret) { exit(ret); }

#define ASSERT(_cond)                                                          \
  do {                                                                         \
    if (!(_cond)) {                                                            \
      printf("ASSERT %s: %d: %s: %s\n", __FILE__, __LINE__, __func__, #_cond); \
      print_trace();                                                           \
      fflush(stdout);                                                          \
      hang(-1);                                                                \
    }                                                                          \
  } while (0)

inline float the_max(float *data, long long count) {
  ASSERT(count > 0);
  ASSERT(data != NULL);
  float a_max = fabs(data[0]);
  for (long long i = 1; i < count; i++) {
    a_max = (a_max < fabs(data[i])) ? fabs(data[i]) : a_max;
  }
  return a_max;
}

inline long long the_min_index(float *data, long long count) {
  ASSERT(data != NULL);
  float a_max = data[0];
  long long min_index = 0;
  for (long long i = 1; i < count; i++) {
    if (a_max > data[i]) {
      a_max = data[i];
      min_index = i;
    }
  }

  return min_index;
}

float real_kl_diversity(float *data, long long count) {
  const long long N = 2048;
  const long long BINS = 128;
  const long long KL_NUM = N / BINS;
  float hist[N] = {0.0};
  float kl[KL_NUM] = {0.0};

  float data_max = the_max(data, count);
  float width = data_max / (N - 1);

  printf("%s %d: data_max=%f width=%f\n", __func__, __LINE__, data_max, width);

  for (long long i = 0; i < count; i++) {
    long long index = floor(fabs(data[i]) / width + 0.5);
    hist[index] += 1;
  }

  long long m = 0;
  float sum = 0;
  for (long long i = BINS; i < N + 1; i += BINS) {
    // P distribution
    float *P = (float *)malloc(sizeof(float) * i);
    for (long long j = 0; j < i; j++) {
      P[j] = 0;
    }
    for (long long j = 0; j < N; j++) {
      long long index = (j > (i - 1)) ? (i - 1) : j;
      P[index] += hist[j];
    }
    for (long long j = 0; j < i; j++) {
      P[j] /= count;
    }

    // Q distribution
    long long expand_size = i / BINS;
    long long idx = 0;
    float *Q = (float *)malloc(sizeof(float) * i);
    for (long long j = i - BINS; j < i; j++) {
      sum += hist[j];
    }
    for (long long j = 0; j < BINS; j++) {
      float sum_bin = 0;
      float positive_cnt = 0;
      long long bin_idx = idx;
      for (long long k = 0; k < expand_size; k++) {
        sum_bin += hist[idx];
        positive_cnt += (hist[idx] > 0) ? 1 : 0;
        idx++;
      }
      positive_cnt = (positive_cnt == 0) ? 1 : positive_cnt;
      float Q_base = sum_bin / positive_cnt / sum;
      while (bin_idx < idx) {
        Q[bin_idx] = hist[bin_idx] ? Q_base : 0;
        bin_idx++;
      }
    }
    for (idx = 0; idx < i; idx++) {
      kl[m] += P[idx] * (log10(P[idx] + 1e-30) - log10(Q[idx] + 1e-30));
    }
    m++;

    free(P);
    free(Q);
  }

  long long m_min = the_min_index(kl, m);
  float threshold = width * (m_min + 1) * BINS;
  printf("  threshold: %.12f, m: %lld, kl: %f\n", threshold, m_min, kl[m_min]);

  return threshold;
}

#ifdef MULTI_THREAD_KL_CALC

void *kl_calc_thread(void *args_input) {
  struct mul_thread_inputs *args;
  args = (struct mul_thread_inputs *)args_input;
  long long *hist = args->hist;
  long long i = args->i;
  long long count = args->count;
  long long N = args->N;
  long long BINS = args->BINS;

  // P distribution
  float *P = (float *)malloc(sizeof(float) * i);
  for (long long j = 0; j < i; j++) {
    P[j] = 0;
  }
  for (long long j = 0; j < N; j++) {
    long long index = (j > (i - 1)) ? (i - 1) : j;
    P[index] += hist[j];
  }
  for (long long j = 0; j < i; j++) {
    P[j] /= count;
  }

  // Q distribution
  float sum = 0.0;
  long long expand_size = i / BINS;
  long long idx = 0;
  float *Q = (float *)malloc(sizeof(float) * i);
  for (long long j = 0; j < i; j++) {
    sum += hist[j];
  }
  for (long long j = 0; j < BINS; j++) {
    float sum_bin = 0;
    float positive_cnt = 0;
    long long bin_idx = idx;
    for (long long k = 0; k < expand_size; k++) {
      sum_bin += hist[idx];
      positive_cnt += (hist[idx] > 0) ? 1 : 0;
      idx++;
    }
    positive_cnt = (positive_cnt == 0) ? 1 : positive_cnt;
    float Q_base = sum_bin / positive_cnt / sum;
    while (bin_idx < idx) {
      Q[bin_idx] = hist[bin_idx] ? Q_base : 0;
      bin_idx++;
    }
  }
  for (idx = 0; idx < i; idx++) {
    *(args->kl) += P[idx] * (log10(P[idx] + 1e-30) - log10(Q[idx] + 1e-30));
  }

  free(P);
  free(Q);

  return NULL;
}

float real_multi_thread_kl_diversity(float *data, long long count,
                                     const long long num_bins) {
  const long long N = num_bins;
  const long long BINS = 128;
  const long long KL_NUM = N / BINS;
  long long *hist = new long long[N];
  float *kl = new float[KL_NUM];

  for (int i = 0; i < N; i++) {
    hist[i] = 0LL;
  }
  for (int i = 0; i < KL_NUM; i++) {
    kl[i] = 0.0;
  }

  float data_max = the_max(data, count);
  float width = data_max / (N - 1);

  printf("%s %d: data_max=%f width=%f\n", __func__, __LINE__, data_max, width);

  for (long long i = 0; i < count; i++) {
    long long index = floor(fabs(data[i]) / width + 0.5);
    hist[index] += 1;
  }

  long long m = 0;
  pthread_t *id = new pthread_t[KL_NUM];
  struct mul_thread_inputs *args_input = new struct mul_thread_inputs[KL_NUM];
  for (long long i = BINS; i < N + 1; i += BINS) {
    args_input[m].hist = hist;
    args_input[m].kl = &kl[m];
    args_input[m].count = count;
    args_input[m].i = i;
    args_input[m].N = N;
    args_input[m].BINS = BINS;
    long long ret =
        pthread_create(&id[m], NULL, kl_calc_thread, (void *)&args_input[m]);
    if (ret) {
      printf("Create No. %lld thread error!\n", m);
      exit(1);
    }
    m++;
  }

  for (long long i = 0; i < m; i++) {
    pthread_join(id[i], NULL);
  }

  long long m_min = the_min_index(kl, m);
  float threshold = width * (m_min + 1) * BINS;
  printf("  threshold: %f, m: %lld, kl: %f\n", threshold, m_min, kl[m_min]);

  delete[] hist;
  delete[] kl;
  delete[] id;
  delete[] args_input;

  return threshold;
}

float real_multi_thread_kl_diversity_hist(int *data, float &width,
                                          const long long N,
                                          const long long BINS) {
  ASSERT(BINS == 128 || BINS == 8);
  const long long KL_NUM = N / BINS;
  long long *hist = new long long[N];
  float *kl = new float[KL_NUM];
  long long count = 0;

  for (int i = 0; i < KL_NUM; i++) {
    kl[i] = 0.0;
  }

  for (int i = 0; i < N; i++) {
    hist[i] = data[i];
    count += hist[i];
  }

  long long m = 0;
  pthread_t *id = new pthread_t[KL_NUM];
  struct mul_thread_inputs *args_input = new struct mul_thread_inputs[KL_NUM];
  for (long long i = BINS; i < N + 1; i += BINS) {
    args_input[m].hist = hist;
    args_input[m].kl = &kl[m];
    args_input[m].count = count;
    args_input[m].i = i;
    args_input[m].N = N;
    args_input[m].BINS = BINS;
    long long ret =
        pthread_create(&id[m], NULL, kl_calc_thread, (void *)&args_input[m]);
    if (ret) {
      printf("Create No. %lld thread error!\n", m);
      exit(1);
    }
    m++;
  }

  for (long long i = 0; i < m; i++) {
    pthread_join(id[i], NULL);
  }

  long long m_min = the_min_index(kl, m);
  float threshold = width * (m_min + 1) * BINS;
  // printf("  threshold: %f, m: %lld, kl: %f, width: %f\n", threshold, m_min,
  // kl[m_min], width);

  delete[] hist;
  delete[] kl;
  delete[] id;
  delete[] args_input;

  return threshold;
}
#endif

float kl_diversity(float *data, long long count, long long num_bins) {
#ifdef MULTI_THREAD_KL_CALC
  return real_multi_thread_kl_diversity(data, count, num_bins);
#else
  return real_kl_diversity(data, count);
#endif
}

float kl_diversity_hist(int *data, float width, long long num_bins,
                        long long dst_bins) {
  return real_multi_thread_kl_diversity_hist(data, width, num_bins, dst_bins);
}
}
