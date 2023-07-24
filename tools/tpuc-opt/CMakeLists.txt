tpu_add_executable(tpuc-opt tpuc-opt.cpp LIBS TpuLibInitAll MLIROptLib MLIRTosaDialect MLIRFuncDialect)
install(TARGETS tpuc-opt DESTINATION bin)
