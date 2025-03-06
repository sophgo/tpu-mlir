#ifndef __PPL_GIT_H__
#define __PPL_GIT_H__
#include <algorithm>
#include <cassert>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif
void ppl_set_node(void *cmdid_node);
int ppl_jit_call(const char *file_name, const char *func_name,
                  const char *args, void *st);
#ifdef __cplusplus
}
#endif
#endif
