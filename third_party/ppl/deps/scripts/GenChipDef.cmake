set(JSON_FILE $ENV{PPL_RUNTIME_PATH}/chip/chip_map_dev.json)
set(CHIP_MAP_H ${CMAKE_BINARY_DIR}/include/chip_map.h)
if(NOT EXISTS "${JSON_FILE}")
    set(JSON_FILE "$ENV{PPL_RUNTIME_PATH}/chip/chip_map.json")
endif()
if(NOT EXISTS "${JSON_FILE}")
    message(FATAL_ERROR "Chip map file not found: ${JSON_FILE}")
endif()

file(READ "${JSON_FILE}" JSON_CONTENT)
set(HEADER_CONTENT
"#pragma once
#include <unordered_map>
#include <string>

static std::unordered_map<std::string, std::string> CHIP_MAP = {
")

string(REGEX MATCHALL "\"([^\"]+)\": \"([^\"]+)\"" PAIRS "${JSON_CONTENT}")
foreach(PAIR IN LISTS PAIRS)
    string(REGEX REPLACE "\"([^\"]+)\": \"([^\"]+)\"" "\\1" KEY "${PAIR}")
    string(REGEX REPLACE "\"([^\"]+)\": \"([^\"]+)\"" "\\2" VALUE "${PAIR}")
    string(APPEND HEADER_CONTENT "    {\"${KEY}\", \"${VALUE}\"},\n")
endforeach()

string(APPEND HEADER_CONTENT "};\n")

file(WRITE "${CHIP_MAP_H}" "${HEADER_CONTENT}")
add_custom_target(GenChipMap DEPENDS ${CHIP_MAP_H})
