# set -ex
SRC=$1
FUNC_NAME=$2
INC=$3
OUT=$4
CHIP=$5
ARG=$6

if [ x${PPL_SRC_PATH} = x ]; then
  PPL_SRC_PATH=$PROJECT_ROOT/ppl_backend/pl/
fi

OPTION_ARG=""
if [ x$ARG != x ]; then
  OPTION_ARG="--const-arg $ARG"
fi

ppl-compile "$SRC"  --I $INC  --x ir --chip $CHIP -D__${CHIP}__ --O3 --desc --device ${OPTION_ARG} -o ${OUT}
ret=$?
if [ $ret != 0 ]; then
  return 1
fi

cp $PPL_PROJECT_ROOT/runtime/scripts/DescDevice.cmake $OUT/CMakeLists.txt
mkdir -p $OUT/build
cd $OUT/build
cmake  ../ -DCHIP=${CHIP} -DNAME=${FUNC_NAME} -DDEBUG=on
make install -j8

ret=$?
if [ $ret != 0 ]; then
  return $ret
else
  return 0
fi
