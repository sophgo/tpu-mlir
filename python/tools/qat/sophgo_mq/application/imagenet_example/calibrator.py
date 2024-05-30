import tensorrt as trt
import os
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image
import numpy as np

def load_imagenet_data(cali_data_loader):
    dataset = []
    for i, (data, label) in enumerate(cali_data_loader):
        data = data.numpy().astype(np.float32)
        dataset.append(data)
    return dataset

class ImagenetCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, cali_data_loader, cache_file):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.cache_file = cache_file

        # Every time get_batch is called, the next batch of size batch_size will be copied to the device and returned.
        self.data = load_imagenet_data(cali_data_loader)
        self.batch_size = self.data[0].shape[0]
        self.current_index = 0

        # Allocate enough memory for a whole batch.
        self.device_input = cuda.mem_alloc(self.data[0].nbytes)

    def get_batch_size(self):
        return self.batch_size

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):
        if self.current_index == len(self.data):
            return None

        batch = self.data[self.current_index].ravel()
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += 1
        print('Calibrate batch = {} / {}'.format(self.current_index, len(self.data)))
        return [self.device_input]


    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        # if os.path.exists(self.cache_file):
        #     with open(self.cache_file, "rb") as f:
        #         return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
