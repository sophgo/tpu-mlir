more info:<https://github.com/google/flatbuffers>

## clone and build github

``` c++
git clone https://github.com/google/flatbuffers.git
cd flatbuffers && git checkout e2be0c0b0605b38e11a8710a8bae38d7f86a7679 && cd ..
mkdir -p flatbuffers/build
cd flatbuffers/build
cmake .. -DCMAKE_INSTALL_PREFIX=install

cmake --build . --target install

```
