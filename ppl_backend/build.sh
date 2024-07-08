set -ex
chmod +x ${PPL_PROJECT_ROOT}/bin/ppl_jit.sh

# clear cache
CACHE_PATH=${PPL_CACHE_PATH}
if [ x${CACHE_PATH} = x ];then
  CACHE_PATH=${HOME}"/.ppl/cache"
fi

rm -rf $CACHE_PATH

mkdir -p build
cd build

for file in `ls ../pl/*.pl`
do
  ppl-compile $file --I $PPL_PROJECT_ROOT/inc  --desc --O2 --o .
done

cmake ../ -DDEBUG=ON
make install -j8
mkdir -p ${PROJECT_ROOT}/install/lib
cp libppl_host.so ${PROJECT_ROOT}/install/lib/
