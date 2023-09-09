# tflite_schema.fbs is tensorflow/tensorflow/lite/schema/schema.fbs@b123bd6284084552fff5f1f9f868f09bb806cd8d
# Tue Mar 15 21:43:01 2022 -0700
execute_process(
  COMMAND flatc
  -o ${CMAKE_CURRENT_SOURCE_DIR} --python ${CMAKE_CURRENT_SOURCE_DIR}/tflite_schema.fbs
)

# fix tflite model
file(GLOB tfliteFiles "./tflite/*.py")
set(match_string "from tflite.")
foreach(eachheaderfile ${tfliteFiles})
  # message(STATUS " INFILES= ${eachheaderfile}\n")
  file(READ ${eachheaderfile} file_content)
  string(REPLACE "${match_string}" "from ." modified_file_content "${file_content}")
  file(WRITE ${eachheaderfile} ${modified_file_content})
  file(READ ${eachheaderfile} file_content1)
endforeach(eachheaderfile)

file(REMOVE "./tflite/__init__.py")
