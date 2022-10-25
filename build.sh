#!/bin/bash
set -e

if [[ -z "$INSTALL_PATH" ]]; then
  echo "Please source envsetup.sh firstly."
  exit 1
fi


BUILD_FLAG="-DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS=-ggdb"


echo "BUILD_PATH: $BUILD_PATH"
echo "INSTALL_PATH: $INSTALL_PATH"

# prepare install/build dir
mkdir -p $BUILD_PATH

pushd $BUILD_PATH
cmake -G Ninja \
    $BUILD_FLAG \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH \
    ${PROJECT_ROOT}
cmake --build . --target install -j8
popd

# Clean up some files for release build
if [ "$1" = "RELEASE" ]; then
  # build doc
  ./release_doc.sh
  # strip mlir tools
  pushd $INSTALL_PATH
  find ./ -name "*.so" |xargs strip
  find ./ -name "*.a" |xargs rm
  popd
fi
