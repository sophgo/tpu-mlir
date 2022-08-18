
## clone and build caffe
# 2022.08.18 8683c89e137ce06ecad3dcb659ed71e1ca352da6
``` c++
git clone git@github.com:sophgo/caffe.git
mkdir -p caffe/build
cd caffe/build
cmake -G Ninja .. \
    -DCPU_ONLY=ON -DUSE_OPENCV=OFF \
    -DBLAS=open -DUSE_OPENMP=TRUE \
    -DCMAKE_CXX_FLAGS=-std=gnu++11 \
    -Dpython_version="3" \
    -DCMAKE_INSTALL_PREFIX=caffe
cmake --build . --target install

copy caffe/python to replace this folder
```

If build failed, you may need to install:
``` c++
apt-get install -y --no-install-recommends \
    libboost-all-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    libhdf5-serial-dev \
    libleveldb-dev \
    liblmdb-dev \
    libprotobuf-dev \
    libsnappy-dev \
    protobuf-compiler \
    libopenblas-dev
```
