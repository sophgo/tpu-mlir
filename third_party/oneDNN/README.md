more info:<https://oneapi-src.github.io/oneDNN/index.html>

## clone and build onednn
# 2022.04.14
``` shell
git clone https://github.com/oneapi-src/oneDNN.git
cd oneDNN && git checkout 20261d29f547e5ff0cb451502fa628672fbd7ed5 && cd ..
mkdir -p oneDNN/build
cd oneDNN/build
cmake .. \
   -DDNNL_CPU_RUNTIME=OMP \
   -DCMAKE_INSTALL_PREFIX=install

cmake --build . --target install -j8

```
