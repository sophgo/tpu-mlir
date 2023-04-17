more info:<https://oneapi-src.github.io/oneDNN/index.html>

## clone and build onednn
# 2023.03.29
``` shell
git clone https://github.com/oneapi-src/oneDNN.git
cd oneDNN && git checkout 5aabea153825347afa92a2d9f69dd893246bea45 && cd ..
mkdir -p oneDNN/build
cd oneDNN/build
cmake .. \
   -DDNNL_CPU_RUNTIME=OMP \
   -DCMAKE_INSTALL_PREFIX=install

cmake --build . --target install -j8

```
