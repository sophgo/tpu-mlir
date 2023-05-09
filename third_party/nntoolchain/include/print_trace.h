#ifndef PRINT_TRACE_
#define PRINT_TRACE_
#ifdef __x86_64__
#include <execinfo.h>
static inline void _print_trace() {
  void* array[10];
  char** strings = NULL;

  int size = backtrace(array, 10);
  strings = backtrace_symbols(array, size);
  printf("Obtained %d stack frames.\n", size);

  for (int i = 0; i < size; i++) {
    printf("%s\n", strings[i]);
  }
  if (strings) free(strings);
}
#else
static inline void _print_trace() {}
#endif

#endif
