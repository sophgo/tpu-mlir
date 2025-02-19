import struct
import numpy as np
try:
    import torch
    use_torch = True
except:
    use_torch = False
import matplotlib.pyplot as plt
import math
import warnings
import sys

use_npz_tool = True
try:
    sys.path.append("..")
    from numpy_helper.tensor_compare import TensorCompare as tc
except:
    warnings.warn("Warning: npz_tool not available in your environment. Fall back to integrated compare.", Warning)
    use_npz_tool = False

from multiprocess import Pool
from pathlib import Path

###########################
# Colored Print Functions #
###########################

def color_str(text: str, color: str | None = None) -> str:
    if color is None or color.lower() == 'none':
        return text
    color_code_dict = {
        'black': 30,
        'red': 31,
        'green': 32,
        'yellow': 33,
        'blue': 34,
        'purple': 35,
        'cyan': 36,
        'white': 37,
    }
    color_code = color_code_dict[color.lower()]
    return f"\033[{color_code}m{text}\033[0m"

###################
# Float Converter #
###################


def fp32_decode(fp32):
    if use_torch:
        if fp32 >> 31:
            fp32 -= 0x100000000
        return torch.tensor(fp32, dtype=torch.int32).view(torch.float32).item()
    return struct.unpack('!f', struct.pack('!I', fp32))[0]


def fp32_encode(flt):
    if use_torch:
        memory = torch.tensor(flt, dtype=torch.float32).view(torch.int32).item()
        if memory < 0:
            memory += 0x100000000
        return memory
    return struct.unpack('!I', struct.pack('!f', flt))[0]


def fp16_decode(fp16):
    if use_torch:
        if fp16 >> 15:
            fp16 -= 0x10000
        return torch.tensor(fp16, dtype=torch.int16).view(torch.float16).item()
    return struct.unpack('<e', struct.pack('<H', fp16))[0]


def fp16_encode(flt):
    if use_torch:
        memory = torch.tensor(flt, dtype=torch.float16).view(torch.int16).item()
        if memory < 0:
            memory += 0x10000
        return memory
    return struct.unpack('<H', np.float16(flt))[0]


def bfp16_decode(bfp16):
    if use_torch:
        if bfp16 >> 15:
            bfp16 -= 0x10000
        return torch.tensor(bfp16, dtype=torch.int16).view(torch.bfloat16).item()
    return struct.unpack('!f', struct.pack('!I', bfp16 << 16))[0]


def bfp16_encode(flt):
    if use_torch:
        memory = torch.tensor(flt, dtype=torch.bfloat16).view(torch.int16).item()
        if memory < 0:
            memory += 0x10000
        return memory
    i = struct.unpack('!I', struct.pack('!f', flt))[0]
    if (i & 0x1FFFF) == 0x8000:
        i -= 0x8000
    i += 0x8000
    return i >> 16


class floating:
    bits = 32

    def __init__(self, value=None, memory=None):
        if memory is None:
            assert not value is None
            self.value = self.decode(self.encode(value))
        else:
            assert value is None
            assert 0 <= memory < (1 << self.bits)
            self.value = self.decode(memory)

    def __repr__(self):
        return f"%s(%s)[0x%0{self.bits // 4}x]" % (self.__class__.__name__, self.value, self.memory)

    def __str__(self):
        return str(float(self))

    def __float__(self):
        return self.value

    def __int__(self):
        return int(self.value)

    def __abs__(self):
        return self.__class__(abs(self.value))

    def __eq__(self, other):
        return self.value == float(other)

    def __lt__(self, other):
        return self.value < float(other)

    def __add__(self, other):
        return self.__class__(self.value + float(other))

    def __radd__(self, other):
        return self.__class__(float(other) + self.value)

    def __sub__(self, other):
        return self.__class__(self.value - float(other))

    def __rsub__(self, other):
        return self.__class__(float(other) - self.value)

    def __mul__(self, other):
        return self.__class__(self.value * float(other))

    def __rmul__(self, other):
        return self.__class__(float(other) * self.value)

    def __truediv__(self, other):
        return self.__class__(self.value / float(other))

    def __rtruediv__(self, other):
        return self.__class__(float(other) / self.value)

    def __pos__(self):
        return self.__class__(self.value)

    def __neg__(self):
        return self.__class__(-self.value)

    @staticmethod
    def decode(memory):
        return fp32_decode(memory)

    @staticmethod
    def encode(flt):
        return fp32_encode(flt)

    @property
    def memory(self):
        return self.encode(self.value)


f32 = floating
fp32 = floating


class float16(floating):
    bits = 16

    def __init__(self, value=None, memory=None):
        super().__init__(value, memory)

    @staticmethod
    def decode(memory):
        return fp16_decode(memory)

    @staticmethod
    def encode(flt):
        return fp16_encode(flt)


f16 = float16
fp16 = float16


class bfloat16(floating):
    bits = 16

    def __init__(self, value=None, memory=None):
        super().__init__(value, memory)

    @staticmethod
    def decode(memory):
        return bfp16_decode(memory)

    @staticmethod
    def encode(flt):
        return bfp16_encode(flt)


bf16 = bfloat16
bfp16 = bfloat16


class FloatConverter:
    def __init__(self, value=None, memory=None):
        if value is None and not memory is None:
            assert isinstance(memory, (int, np.integer))
        elif not value is None and memory is None:
            assert isinstance(value, (float, np.floating, int, np.integer))
            warnings.warn(
                    "Warning: You are inputting an integer as value. If it is not intended, specify 'memory=' argument.", Warning)
        else:
            raise ValueError
        self.memory = memory
        self.value = value
        self.init_memory()
        self.init_value()

    def init_memory(self):
        if not self.memory is None:
            self.fp32_memory = self.memory & 0xffffffff
            self.bfp16_memory = self.memory & 0xffff
            self.fp16_memory = self.memory & 0xffff

        if not self.value is None:
            self.fp32_memory = fp32_encode(self.value)
            self.bfp16_memory = bfp16_encode(self.value)
            self.fp16_memory = fp16_encode(self.value)

    def init_value(self):
        self.fp32 = fp32_decode(self.fp32_memory)
        self.bfp16 = bfp16_decode(self.bfp16_memory)
        self.fp16 = fp16_decode(self.fp16_memory)

    def print_results(self):
        print("BF16: %10d     0x%04x" %
              (self.bfp16_memory, self.bfp16_memory), "Value:", self.bfp16, )
        print("FP16: %10d     0x%04x" %
              (self.fp16_memory, self.fp16_memory), "Value:", self.fp16, )
        print("FP32: %10d 0x%08x" %
              (self.fp32_memory, self.fp32_memory), "Value:", self.fp32, )


##########################
# NPZ Compare Visualizer #
##########################

default_tol = {'f32': {'abs_tol': 2 ** -126, 'rel_tol': 2 ** -23},
               'f16': {'abs_tol': 2 ** -14, 'rel_tol': 2 ** -10},
               'bf16': {'abs_tol': 2 ** -126, 'rel_tol': 2 ** -7},
               'int8': {'abs_tol': 0., 'rel_tol': 0.}}

class NPZTdbErrWrapper:
    def __init__(self, npz, role):
        assert isinstance(npz, np.lib.npyio.NpzFile)
        assert role in ['desired', 'actual']
        self.role = role
        self.npz = npz
        self._valid_keys = None

    @property
    def valid_keys(self):
        if self._valid_keys is None:
            self._valid_keys = {name[:-len(self.role)-1] for name in self.npz.keys() if name.endswith(self.role)}
        assert isinstance(self._valid_keys, set)
        return self._valid_keys

    def keys(self):
        return self.valid_keys

    def values(self):
        return (f"{name}_{self.role}" for name in self._valid_keys)

    def items(self):
        return zip(self.valid_keys, self.values())

    def __contains__(self, item):
        return item in self.valid_keys

    def __iter__(self):
        return iter(self.valid_keys())

    def __getitem__(self, key):
        return self.npz[f"{key}_{self.role}"]

    def __setitem__(self, key, value):
        self.npz[f"{key}_{self.role}"] = value

    def __len__(self):
        return len(self.valid_keys)

def tdb_err_comparer(fn):
    if isinstance(fn, str):
        npz = np.load(fn)
    elif isinstance(fn, np.lib.npyio.NpzFile):
        npz = fn
    else:
        assert 0, "Should provide file name or npz object of tdb_err.npz!"
    target = NPZTdbErrWrapper(npz, 'actual')
    ref = NPZTdbErrWrapper(npz, 'desired')
    return NPZComparer(target, ref)


def model_tpu_comparer(fn):
    assert isinstance(fn, str) and fn.endswith(".npz"), "Please provide the file name of npz file!"
    if "_model_" in fn:
        pattern = fn.replace("_model_", "%s")
    elif "_tpu_" in fn:
        pattern = fn.replace("_tpu_", "%s")
    else:
        assert 0, "Should provide either model_out or tpu_out npz file!"
    return NPZComparer(pattern % "_model_", pattern % "_tpu_")


def get_nchw(darray, mix_axis):
    shape = darray.shape
    dims = len(shape)
    reshaped = [1, 1, 1, 1]
    if mix_axis is None:
        if dims > 4:
            for i in range(2):
                reshaped[- i - 1] = shape[- i - 1]
            for i in range(1, dims - 2):
                reshaped[1] *= shape[i]
            reshaped[0] = shape[0]
        else:
            for i in range(dims):
                reshaped[- i - 1] = shape[- i - 1]
    else:
        assert dims - len(mix_axis) + 1 == 4
        dim = mix_axis[0]
        for i in range(1, len(mix_axis)):
            assert mix_axis[i] - mix_axis[i - 1] == 1
        for i in mix_axis:
            reshaped[dim] *= shape[i]
        for i in range(dim):
            reshaped[i] = shape[i]
        for i in range(dims - dim - len(mix_axis)):
            reshaped[3 - i] = shape[dims - i - 1]
    return reshaped


def recalc_hw(h, w):
    all_hw = h * w
    for i in range(int(math.sqrt(all_hw)), 0, -1):
        if all_hw % i == 0:
            return all_hw // i, i


def assign_hw(h, w, new_hw):
    n_ele = h * w
    h1, w1 = new_hw
    if h1 == -1 and w1 == -1:
        return math.ceil(np.sqrt(n_ele)), math.ceil(np.sqrt(n_ele))
    elif h1 == -1 and w1 != -1:
        return math.ceil(n_ele / w1), w1
    elif h1 != -1 and w1 == -1:
        return h1, math.ceil(n_ele / h1)
    else:
        assert h1 * w1 >= n_ele
        return h1, w1


def assign_new_shape(reshaped, resize_hw):
    h, w = reshaped[2], reshaped[3]
    if isinstance(resize_hw, str):
        if resize_hw.lower() == "rectangle" or resize_hw.lower() == "auto":
            reshaped[2], reshaped[3] = recalc_hw(h, w)
        elif resize_hw.lower() == 'square':
            reshaped[2], reshaped[3] = assign_hw(h, w, (-1, -1))
        elif resize_hw.lower() == "none":
            pass
        else:
            assert 0, "Invalid param resize_hw=%s" % resize_hw
    elif isinstance(resize_hw, int):
        assert resize_hw > 0
        reshaped[2], reshaped[3] = assign_hw(h, w, (resize_hw, -1))
    elif isinstance(resize_hw, (tuple, list)):
        assert len(resize_hw) == 2
        reshaped[2], reshaped[3] = assign_hw(h, w, resize_hw)
    elif resize_hw is None:
        pass
    else:
        assert 0, "Invalid param resize_hw=%s" % resize_hw
    return np.array([1] * (h * w) + [0] * (reshaped[2] * reshaped[3] - h * w)).reshape((reshaped[2], reshaped[3]))


def get_data_dist(darray, data_mask):
    to_calculate = darray[data_mask == 1]
    to_calculate = to_calculate[~np.isinf(to_calculate)]
    real_mean = np.nanmean(to_calculate)
    real_min = np.nanmin(to_calculate)
    real_max = np.nanmax(to_calculate)
    return real_mean, real_min, real_max


def plot_2d_array(diff, data_mask=None, title="", figsize=6, vmin=-0.1, vmax=0.1, h_split=None, w_split=None):
    figwidth = figsize
    figheight = 3 + figsize / diff.shape[1] * diff.shape[0]
    plt.figure(figsize=(figwidth, figheight))
    nan_mask = np.where(np.isnan(diff).astype(float) * data_mask, 1, np.nan)
    pos_inf_mask = np.where(np.isposinf(diff).astype(float) * data_mask, 1, np.nan)
    neg_inf_mask = np.where(np.isneginf(diff).astype(float) * data_mask, 1, np.nan)
    plt.imshow(nan_mask, 'Greens', vmin=0, vmax=1.25)
    plt.imshow(pos_inf_mask, 'Oranges', vmin=0, vmax=1.25)
    plt.imshow(neg_inf_mask, 'Greys', vmin=0, vmax=1.25)
    if not data_mask is None:
        # diff += np.where(data_mask == 0, np.nan, 0)
        diff[data_mask == 0] += np.nan
    plt.imshow(diff, 'bwr', vmin=vmin, vmax=vmax)
    plt.xlim(-2, diff.shape[1] + 1)
    plt.ylim(diff.shape[0] + 1, -2)
    ax = plt.gca()
    ax.set_facecolor('lightgrey')
    ax.xaxis.set_tick_params(which='major', top=True,
                             labeltop=True, labelbottom=True)
    ax.yaxis.set_tick_params(which='major', right=True,
                             labelleft=True, labelright=True)

    if h_split is None:
        h_split = max(1, int(diff.shape[0] / (figheight - 3) / 3 + 0.5))
        h_ticks = np.arange(0, diff.shape[0], h_split)
        h_range = range(0, diff.shape[0], h_split)
    else:
        h_ticks = np.arange((h_split - 1) / 2, diff.shape[0], h_split)
        h_range = range(diff.shape[0] // h_split)
    if w_split is None:
        w_split = max(1, int(diff.shape[1] / figwidth / 3 + 0.5))
        w_ticks = np.arange(0, diff.shape[1], w_split)
        w_range = range(0, diff.shape[1], w_split)
        rotation = 'vertical' if diff.shape[1] > 100 + w_split else None
    else:
        w_ticks = np.arange((w_split - 1) / 2, diff.shape[1], w_split)
        w_range = range(diff.shape[1] // w_split)
        rotation = 'vertical' if diff.shape[1] // w_split >= 100 else None
    plt.yticks(h_ticks, h_range)
    for i in range(h_split, diff.shape[0], h_split):
        plt.axhline(i - 0.5, 0, diff.shape[1], color='lightgrey')
    plt.xticks(w_ticks, w_range, rotation=rotation)
    for i in range(w_split, diff.shape[1], w_split):
        plt.axvline(i - 0.5, 0, diff.shape[0], color='lightgrey')
    title_len, title_line = len(title), figsize * 8
    plt.title(title if title_len <= title_line else "\n".join(
        [title[i: i + title_line] for i in range(0, title_len, title_line)]))
    # plt.colorbar()
    plt.show()


def make_slice_object(something):
    if isinstance(something, slice):
        return something
    elif isinstance(something, int):
        if something == -1:
            return slice(None)
        return slice(something, something + 1)
    elif isinstance(something, (list, tuple)):
        assert 1 <= len(something) <= 3
        return slice(*something)
    elif something is None:
        return None
    else:
        raise ValueError("Invalid slice param")


class NPZWrapper:
    def __init__(self, npz, role="darray"):
        if isinstance(npz, (str, Path)):
            self.npz = np.load(npz)
        elif isinstance(npz, (np.lib.npyio.NpzFile, dict, NPZTdbErrWrapper)):
            self.npz = npz
        elif isinstance(npz, np.ndarray):
            self.npz = {'darray': npz}
        else:
            raise TypeError
        self.role = role

    def keys(self):
        return self.npz.keys()

    def values(self):
        return self.npz.values()

    def items(self):
        return self.npz.items()

    def __contains__(self, item):
        return item in self.npz

    def __iter__(self):
        return iter(self.keys())

    def __getitem__(self, key):
        return self.npz[key]

    def __setitem__(self, key, value):
        self.npz[key] = value

    def __len__(self):
        return len(self.npz)

    def info(self, tensor=None):
        if tensor is None:
            tensors = self.keys()
        else:
            if isinstance(tensor, list):
                for ts in tensor:
                    assert ts in self.keys()
                tensors = tensor
            else:
                tensors = [tensor]
        for key in tensors:
            print("tensor='%s'," % key, "#shape", self[key].shape)

    def get_darray(self, tensor=None, slices=None, index=None, c_columns=32, resize_hw=None, transpose_hw=False, mix_axis=None, print_shape=True):
        if tensor is None:
            tensor = list(self.keys())[0]
        slice_list = tuple(make_slice_object(e)
                           for e in slices) if not slices is None else (slice(None),)
        if len(self[tensor].shape) == 0:
            darray = self[tensor].reshape(1).astype(float)
        else:
            darray = self[tensor][slice_list].astype(float)
        reshaped = get_nchw(darray, mix_axis)
        if print_shape:
            if not slices is None:
                print("sliced", end=" ")
            print("shape %s, reshaped to %s" %
                  (darray.shape, tuple(reshaped)), end=", ")
        darray = np.reshape(darray, reshaped)
        data_mask_channel = assign_new_shape(reshaped, resize_hw)
        if transpose_hw:
            darray = np.transpose(darray, [0, 1, 3, 2])
            if resize_hw in {'none', None, 'rectangle', 'auto'}:
                reshaped[2], reshaped[3] = reshaped[3], reshaped[2]
                data_mask_channel = np.reshape(
                    data_mask_channel.reshape(-1), data_mask_channel.shape[::-1])
            if print_shape:
                print("with hw transposed data", end=" ")
        if print_shape:
            print("shown in %s" % (tuple(reshaped), ))

        n_, c_, h_, w_ = reshaped
        per_c = math.ceil(c_ / c_columns)

        if not index is None:
            n, c = index
            new_darray = np.resize(
                darray[n // per_c][(n % per_c) * c_columns + c], data_mask_channel.shape)
            data_mask = data_mask_channel
            h_, w_ = None, None
        else:
            frame_shape = [n_ * per_c, min(c_, c_columns)]
            new_darray = np.block([[np.resize(darray[n // per_c][(n % per_c) * c_columns + c], data_mask_channel.shape) if 0 <= (n % per_c) *
                                    c_columns + c < c_ else np.zeros_like(data_mask_channel) for c in range(frame_shape[1])] for n in range(frame_shape[0])])
            data_mask = np.block([[data_mask_channel if 0 <= (n % per_c) *
                                   c_columns + c < c_ else np.zeros_like(data_mask_channel) for c in range(frame_shape[1])] for n in range(frame_shape[0])])

        attr = {'title': ' '.join([tensor, str(slices) if not slices is None else "", str(index) if not index is None else ""]).strip(),
                'h_split': h_,
                'w_split': w_,
                }

        return new_darray, data_mask, attr

    def plot(self, tensor=None, abs_tol=None, rel_tol=None, figsize=6, vmin=None, vmax=None, **kwargs):
        if tensor is None:
            warnings.warn(
                "Your are plotting all the tensors in the NPZ file. This may cause problems when the file is large.", Warning)
            tensors = self.keys()
        else:
            if isinstance(tensor, list):
                for ts in tensor:
                    assert ts in self.keys()
                tensors = tensor
            else:
                tensors = [tensor]

        for key in tensors:
            print("tensor='%s'," % key)
            darray, data_mask, attr = self.get_darray(key, **kwargs)
            real_mean, real_min, real_max = get_data_dist(darray, data_mask)
            print("data distribution: mean %s, min %s, max %s" %
                  (real_mean, real_min, real_max))

            if abs_tol is None:
                abs_tol_ = real_mean
            else:
                abs_tol_ = abs_tol
            if rel_tol is None:
                rel_tol_ = max(abs(real_max - abs_tol_),
                               abs(real_min - abs_tol_))
            else:
                rel_tol_ = rel_tol
            if rel_tol_ == 0.:
                rel_tol_ += 1E-10
            if vmin is None or vmax is None:
                vmin = -1  # (real_min - abs_tol) / rel_tol
                vmax = 1  # (real_max - abs_tol) / rel_tol
            # print("zero point %s, scale %s." % (abs_tol_, rel_tol_))
            print(f"vmin {abs_tol_ - rel_tol_} zero point {abs_tol_} vmax {abs_tol_ + rel_tol_} ")

            attr['title'] = "%s: %s" % (self.role, attr['title'])
            plot_2d_array((darray - abs_tol_) / rel_tol_, data_mask, figsize=figsize,
                          vmin=vmin, vmax=vmax, **attr)


class NPZComparer:
    def __init__(self, target, ref):
        self.target = NPZWrapper(target, 'Target')
        self.ref = NPZWrapper(ref, 'Ref')
        self._keys = None
        self._error_keys = None
        self.archived_kwargs = {}

    # lazy initialization
    @property
    def keys(self):
        if self._keys is None and self._error_keys is None:
            self._keys = {}
            self._error_keys = {}
            for key in self.ref:
                if key in self.target:
                    if self.target[key].size != self.ref[key].size:
                        print(f"Error: Tensor {key} shape not same: {self.target[key].shape} vs {self.ref[key].shape}.")
                        self._error_keys[key] = 1
                    else:
                        self._keys[key] = 1
            assert len(self._keys) > 0, 'No common data.'
        assert(isinstance(self._keys, dict) and isinstance(self._error_keys, dict)), "Keys not initialized!"
        return self._keys

    @property
    def error_keys(self):
        if self._keys is None and self._error_keys is None:
            self.keys
        assert(isinstance(self._keys, dict) and isinstance(self._error_keys, dict)), "Keys not initialized!"
        return self._error_keys

    def info(self, tensor=None):
        if tensor is None:
            tensors = self.keys
            print(f"Target tensors: {len(self.target)}, Ref tensors: {len(self.ref)}, Common tensors: {len(self.keys)}, Unmatched tensors:{len(self.error_keys)}")
        else:
            if isinstance(tensor, list):
                for ts in tensor:
                    assert ts in self.keys
                tensors = tensor
            else:
                tensors = [tensor]
        for key in tensors:
            print("tensor='%s'," % key, "#shape", self.ref[key].shape)

    def compare(self, tolerance=(-1., -1.), tensor=None, verbose=True, summary=False):
        if tensor is None:
            tensors = self.keys
            if verbose:
                print(f"Target tensors: {len(self.target)}, Ref tensors: {len(self.ref)}, Common tensors: {len(self.keys)}, Unmatched tensors:{len(self.error_keys)}")
        else:
            if isinstance(tensor, list):
                for ts in tensor:
                    assert ts in self.keys
                tensors = tensor
            else:
                tensors = [tensor]

        min_similarity = np.array([1.0, 1.0, np.inf])
        ALL_PASS = 1

        results = {}
        with Pool() as pool:
            for key in tensors:
                results[key] = pool.apply_async(calc_similarity, args=(self.target[key].reshape(self.ref[key].shape), self.ref[key]))
            pool.close()
            pool.join()

        for key in tensors:
            result = results[key].get()
            PASS = 1
            if 'similarity' in result:
                similarity = np.array(result['similarity'])
                min_similarity = np.minimum(min_similarity, similarity)
                if np.any(similarity[:2] < np.array(tolerance)):
                    PASS = 0
                    ALL_PASS = 0
            if verbose and not summary:
                print(color_str(f"tensor='{key}', #{self.ref[key].shape} {''.join(('%s: (%.6f, %.6f, %.6f)' % (k, *v) if isinstance(v, tuple) else '%s: %s' % (k, v)) for k, v in result.items())}{(' √' if PASS else ' ×') if tolerance != (-1., -1) else '' }", 'none' if tolerance == (-1., -1) else ('green' if PASS else 'red')))

        if verbose:
            print(color_str(f"min_similarity: {tuple(min_similarity)}{(' √' if ALL_PASS else ' ×') if tolerance != (-1., -1.) else ''}", 'none' if tolerance == (-1., -1.) else ('green' if ALL_PASS else 'red')))
        return bool(ALL_PASS)

    def get_diff(self, tensor=None, abs_tol=0, rel_tol=0, **kwargs):
        if tensor is None:
            tensor = list(self.keys())[0]
        target, data_mask1, attr1 = self.target.get_darray(
            tensor, print_shape=False, **kwargs)
        ref, data_mask2, attr2 = self.ref.get_darray(
            tensor, print_shape=False, **kwargs)
        if target.shape != ref.shape:
            assert target.size == ref.size
            target = target.reshape(ref.shape)
            data_mask1 = data_mask1.reshape(data_mask2.shape)
            attr1 = attr2
        assert target.shape == ref.shape and np.all(
            data_mask1 == data_mask2) and attr1 == attr2
        compare = calc_similarity(target, ref, data_mask1)

        mask = 1 - np.isclose(target, ref, atol=abs_tol, rtol=rel_tol)
        if np.sum(mask) == 0:
            attr1['title'] += ' - No difference'
            warnings.warn("No difference under given tolerances.")

        diff = target - ref
        diff *= mask
        if rel_tol != 0:
            diff /= np.abs(ref + 1E-10)
        attr1['title'] = 'Diff: %s' % attr1['title']

        return diff, data_mask1, attr1, compare

    def plot_diff(self, tensor=None, abs_tol=0, rel_tol=0, figsize=6, vmin=None, vmax=None, **kwargs):
        if tensor is None:
            warnings.warn(
                "Your are plotting all the tensors in the NPZ file. This may cause problems when the file is large.", Warning)
            tensors = self.keys
        else:
            if isinstance(tensor, list):
                for ts in tensor:
                    assert ts in self.keys
                tensors = tensor
            else:
                tensors = [tensor]
        print("abs tol %s, rel tol %s" % (abs_tol, rel_tol))
        for key in tensors:
            print("tensor='%s'," % key)
            darray, data_mask, attr, compare = self.get_diff(
                key, abs_tol=abs_tol, rel_tol=rel_tol,  **kwargs)
            real_mean, real_min, real_max = get_data_dist(darray, data_mask)
            abs_mean, abs_min, abs_max = get_data_dist(np.abs(darray), data_mask)
            print("max diff neg %s, pos %s, mean %s\nabs diff min %s, max %s, mean %s" % (real_min, real_max, real_mean, abs_min, abs_max, abs_mean))
            print(*(f"{k}: {v}" for k, v in compare.items()))
            if vmin is None or vmax is None:
                diff_max = max(abs(real_min), abs(real_max))
                if diff_max == 0:
                    diff_max += 1
                vmin_ = - diff_max
                vmax_ = diff_max
            else:
                vmin_ = vmin
                vmax_ = vmax
            print("vmin %s vmax %s" % (vmin_, vmax_))
            plot_2d_array(darray, data_mask, figsize=figsize,
                          vmin=vmin_, vmax=vmax_, **attr)

    def plot(self, *args, **kwargs):
        self.plot_diff(*args, **kwargs)

    def plot_vs(self, tensor=None, abs_tol=0, rel_tol=0, figsize=6, vmin=None, vmax=None, zero_point=None,
                slices=None, index=None, c_columns=32, resize_hw=None, transpose_hw=False, mix_axis=None,
                dump=False, verbose=False):
        # archive the kwargs for next dump_vs
        self.archived_kwargs = {}
        self.archived_kwargs.update(
            tensor=tensor, abs_tol=abs_tol, rel_tol=rel_tol,
            slices=slices, index=index, c_columns=c_columns, resize_hw=resize_hw, transpose_hw=transpose_hw, mix_axis=mix_axis,
        )

        kwargs = {'slices': slices, 'index': index, 'c_columns': c_columns, 'resize_hw': resize_hw, 'transpose_hw': transpose_hw, 'mix_axis': mix_axis}
        tensors = []
        if tensor is None:
            warnings.warn(
                "Your are plotting all the tensors in the NPZ file. This may cause problems when the file is large.", Warning)
            tensors.extend(self.keys)
        else:
            if isinstance(tensor, list):
                for tensor in tensor:
                    assert tensor in self.keys
                    tensors.append(tensor)
            else:
                tensors.append(tensor)
        if abs_tol != 0 or rel_tol != 0 or not vmin is None or not vmax is None:
            warnings.warn(
                "In plot_vs, abs_tol, rel_tol, vmin, vmax only affect plot_diff. Set zero_point to affect plot_ref and plot_target.", Warning)
        for key in tensors:
            target_darray, data_mask1, attr1 = self.target.get_darray(
                key, print_shape=False, **kwargs)
            ref_darray, data_mask2, attr2 = self.ref.get_darray(
                key, print_shape=False, **kwargs)
            assert np.all(data_mask1 == data_mask2) and attr1 == attr2
            target_mean, target_min, target_max = get_data_dist(
                target_darray, data_mask1)
            ref_mean, ref_min, ref_max = get_data_dist(ref_darray, data_mask2)
            zp = zero_point if not zero_point is None else (target_mean + ref_mean) / 2
            all_min = min(target_min, ref_min)
            all_max = max(target_max, ref_max)
            scale = max(abs(all_max - zp), abs(all_min - zp))
            if scale == 0:
                scale += 1E-10
            all_vmin = -1  # (all_min - zp) / scale
            all_vmax = 1  # (all_max - zp) / scale

            self.target.plot(key, abs_tol=zp, rel_tol=scale,
                             figsize=figsize, vmin=all_vmin, vmax=all_vmax, **kwargs)
            self.ref.plot(key, abs_tol=zp, rel_tol=scale,
                          figsize=figsize, vmin=all_vmin, vmax=all_vmax, **kwargs)
            self.plot_diff(key, abs_tol=abs_tol, rel_tol=rel_tol,
                           figsize=figsize, vmin=vmin, vmax=vmax, **kwargs)
        if dump:
            self.dump_vs_plot(verbose=verbose)

    def plot_ref(self, *args, **kwargs):
        self.ref.plot(*args, **kwargs)

    def plot_target(self, *args, **kwargs):
        self.target.plot(*args, **kwargs)

    def dump_vs_plot(self, abs_tol=None, rel_tol=None, verbose=None, **kwargs):
        archived_kwargs = self.archived_kwargs.copy()
        archived_kwargs.update(kwargs)
        if not abs_tol is None:
            archived_kwargs['abs_tol'] = abs_tol
        if not rel_tol is None:
            archived_kwargs['rel_tol'] = rel_tol
        if not verbose is None:
            archived_kwargs['verbose'] = verbose
        self.dump_vs(**archived_kwargs)

    def dump_vs(self, tensor=None, abs_tol=1e-8, rel_tol=1e-3,
                slices=None, index=None, c_columns=32, resize_hw=None, transpose_hw=False, mix_axis=None,
                verbose=False, **kwargs):
        kwargs_plot = {'slices': slices, 'index': index, 'c_columns': c_columns, 'resize_hw': resize_hw, 'transpose_hw': transpose_hw, 'mix_axis': mix_axis}
        tensors = []
        if tensor is None:
            warnings.warn(
                "Your are dumping all the tensors in the NPZ file. This may cause problems when the file is large.", Warning)
            tensors.extend(self.keys)
        else:
            if isinstance(tensor, list):
                for tensor in tensor:
                    assert tensor in self.keys
                    tensors.append(tensor)
            else:
                tensors.append(tensor)
        for key in tensors:
            print(f"tensor='{key}', # original shape {self.ref[key].shape}", end=", ")
            target_darray, data_mask1, attr1 = self.target.get_darray(
                key, print_shape=False, **kwargs_plot)
            ref_darray, data_mask2, attr2 = self.ref.get_darray(
                key, print_shape=True, **kwargs_plot)
            assert np.all(data_mask1 == data_mask2) and attr1 == attr2

            print(f"{''.join(('%s: (%.6f, %.6f, %.6f)' % (k, *v) if isinstance(v, tuple) else '%s: %s' % (k, v)) for k, v in calc_similarity(target_darray, ref_darray, data_mask1).items())}", end=", ")
            print("tolerance: abs %s, rel %s" % (abs_tol, rel_tol))

            diff = target_darray - ref_darray
            rel_diff = diff / np.abs(ref_darray + 1e-10)
            print("%20s %15s %15s %15s %15s" % ("index", "target", "ref", "diff", "rel_diff"))

            N, C, H, W = idx_to_nchw(np.array(target_darray.shape) - 1, attr1)
            for n in range(N+1):
                for c in range(C+1):
                    for h in range(H+1):
                        for w in range(W+1):
                            h_, w_ = nchw_to_idx((n, c, h, w), attr1)
                            if not data_mask1[h_][w_]:
                                continue
                            error_mark_abs = int(min(100, abs(diff[h_][w_]) / (abs_tol + 1e-10)))
                            error_mark_rel = int(min(100, abs(rel_diff[h_][w_]) / (rel_tol + 1e-10)))
                            error_mark = min(error_mark_abs, error_mark_rel)
                            if verbose or error_mark >= 1:
                                color = 'none' if diff[h_][w_] == 0 else ('blue' if diff[h_][w_] < 0 else 'red')
                                print("%20s %#15.8g %#15.8g %s %s %s" % ((h, w) if attr1["h_split"] is None else (n, c, h, w),
                                                                         target_darray[h_][w_],
                                                                         ref_darray[h_][w_],
                                                                         color_str("%#15.8g" % diff[h_][w_], 'green' if abs(diff[h_][w_]) <= abs_tol else color),
                                                                         color_str("%#15.8g" % rel_diff[h_][w_], 'green' if abs(rel_diff[h_][w_]) <= rel_tol else color),
                                                                         color_str("!" * error_mark, color)))
            print("")

def idx_to_nchw(idx, attr):
    if attr['w_split'] is None and attr['h_split'] is None:
        return (0, 0, *idx)
    n = idx[0] // attr['h_split']
    c = idx[1] // attr['w_split']
    h = idx[0] % attr['h_split']
    w = idx[1] % attr['w_split']
    return (n, c, h, w)

def nchw_to_idx(nchw, attr):
    n, c, h, w = nchw
    if attr['w_split'] is None and attr['h_split'] is None:
        return (h, w)
    return (n * attr['h_split'] + h, c * attr['w_split'] + w)

def calc_similarity(target, ref, mask=None):
    if mask is None:
        mask = np.ones_like(target)
    target_flatten = target[mask == 1].flatten().astype(float)
    ref_flatten = ref[mask == 1].flatten().astype(float)
    # compare similarity as npz_tool does
    if use_npz_tool:
        mlir_comparer = tc(euclidean_similarity_tol=float('-inf'))
        _1, level, _2, sim, _3 = mlir_comparer.compare(target_flatten, ref_flatten, False, False)
        if level == tc.EQUAL:
            return {"equal": "all equal"}
        if level == tc.CLOSE:
            return {"close_order": sim["close_order"]}
        if level == tc.SIMILAR or level == tc.NOT_SIMILAR:
            return {"similarity": (sim["cosine"], sim["euclid"], sim["sqnr"])}
        raise ValueError("Invalid level %s" % level)
    else:
        def square_rooted(x):
            return np.sqrt(np.sum(np.power(x, 2)))

        def cosine_similarity(x, y):
            numerator = np.sum(x * y)
            sqrt_x = square_rooted(x)
            sqrt_y = square_rooted(y)
            denominator = sqrt_x * sqrt_y
            if denominator == 0.0:
                if sqrt_x == 0.0 and sqrt_y == 0.0:
                    return 1.0
                else:
                    return 0.0
            return numerator / denominator

        def euclidean_similarity(x, y):
            ed = np.sqrt(np.sum(np.power(x - y, 2)))
            sr = square_rooted((x + y) / 2) + 1e-7
            if (np.isinf(ed) or np.isinf(sr)):
                res = 0.0
            else:
                res = 1 - ed / sr
            return res

        def sqnr_similarity(signal_raw, signal_dequant):
            raw = signal_raw.ravel()
            dequant = signal_dequant.ravel()

            noise = raw - dequant

            avg_raw = np.sum(raw) / raw.size
            avg_noise = np.sum(noise) / noise.size

            var_raw_zero_mean = np.sum(np.square(raw - avg_raw))
            var_noise_zero_mean = np.sum(np.square(noise - avg_noise))
            if var_noise_zero_mean == 0 or var_raw_zero_mean == 0:
                return float('inf')
            sqnr = 10 * np.log10(var_raw_zero_mean / var_noise_zero_mean)

            return sqnr

        def close_order(x, y):
            for order in range(5, 1, -1):
                if (np.allclose(x, y, rtol=1 * 10**(-order), atol=1e-8,
                                equal_nan=True)):
                    return order
            return 0

        if np.all(target_flatten == ref_flatten):
            return {"equal": "all equal"}

        close = close_order(target_flatten, ref_flatten)
        if close >= 3:
            return {"close_order": close}

        target_flatten[np.isnan(target_flatten)] = 0.0
        ref_flatten[np.isnan(ref_flatten)] = 0.0
        target_flatten[np.isposinf(target_flatten)] = 10000.0
        target_flatten[np.isneginf(target_flatten)] = -10000.0
        ref_flatten[np.isposinf(ref_flatten)] = 10000.0
        ref_flatten[np.isneginf(ref_flatten)] = -10000.0

        cos_sim = cosine_similarity(target_flatten, ref_flatten)
        euc_sim = euclidean_similarity(target_flatten, ref_flatten)
        sqnr_sim = sqnr_similarity(target_flatten, ref_flatten)
        return {"similarity": (cos_sim, euc_sim, sqnr_sim)}
