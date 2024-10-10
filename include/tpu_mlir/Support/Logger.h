#pragma once
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdarg>
#include <cstdio>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

using namespace llvm;

namespace tpu_mlir {
extern int32_t cur_log_level;

inline std::string formatString(const char *format, ...) {
  va_list args;
  va_start(args, format);
  size_t size = vsnprintf(nullptr, 0, format, args) + 1; // Extra space for '\0'
  std::unique_ptr<char[]> buf(new char[size]);
  vsnprintf(buf.get(), size, format, args);
  va_end(args);
  return std::string(buf.get(),
                     buf.get() + size - 1); // We don't want the '\0' inside
}

#define PASS_LOG_DEBUG_BLOCK(block)                                            \
  do {                                                                         \
    if (cur_log_level == 1 || cur_log_level == 0) {                            \
      std::string log_output;                                                  \
      llvm::raw_string_ostream rso(log_output);                                \
      block;                                                                   \
      llvm::outs() << rso.str() << "\n";                                       \
    }                                                                          \
  } while (0)

#define LAYER_GROUP_LOG_DEBUG_BLOCK(block)                                     \
  do {                                                                         \
    if (cur_log_level == 2 || cur_log_level == 0) {                            \
      std::string log_output;                                                  \
      llvm::raw_string_ostream rso(log_output);                                \
      block;                                                                   \
      llvm::outs() << rso.str();                                               \
    }                                                                          \
  } while (0)

inline void SetLogFlag(int32_t log_level) { cur_log_level = log_level; }

#define PROFILE_LOG(step, begin)                                               \
  do {                                                                         \
    DEBUG_WITH_TYPE("profile", {                                               \
      auto current_time = std::chrono::high_resolution_clock::now();           \
      auto time_string = std::chrono::system_clock::to_time_t(current_time);   \
      if (begin) {                                                             \
        llvm::dbgs() << "; action = profile" << "; step = " << step            \
                     << "; begin = " << std::ctime(&time_string) << "\n";      \
      } else {                                                                 \
        llvm::dbgs() << "; action = profile" << "; step = " << step            \
                     << "; end = " << std::ctime(&time_string) << "\n";        \
      }                                                                        \
    });                                                                        \
  } while (0)

} // namespace tpu_mlir
