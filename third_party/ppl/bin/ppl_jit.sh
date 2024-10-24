# set -ex
SRC=$1
FUNC_NAME=$2
INC=$3
OUT=$4
CHIP=$5
ARG=$6

OPTION_ARG=""
if [ x$ARG != x ]; then
  OPTION_ARG="--const-arg $ARG"
fi

ppl-compile "$SRC"  --I $INC  --x ir --chip $CHIP -D__${CHIP}__ --O3 --desc --device ${OPTION_ARG} -o ${OUT}
ret=$?
if [ $ret -eq 0x1001 ]; then
    echo "Error: Address assign failed!"
    return 0x1001
elif [ $ret -ne 0 ]; then
    return $ret
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
