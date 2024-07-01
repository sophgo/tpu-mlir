import numpy as np
import struct

equal_dtypes = {
    "i4": "int4",
    "u4": "uint4",
    "ui4": "uint4",
    "i8": "int8",
    "u8": "uint8",
    "ui8": "uint8",
    "si8": "int8",
    "i16": "int16",
    "u16": "uint16",
    "i32": "int32",
    "i64": "int64",
    "ui16": "uint16",
    "i32": "int32",
    "ui32": "uint32",
    "f16": "float16",
    "f32": "float32",
}


def lowering(input:np.array, pdtype:str, pshape, pzero_point=0, pscale=1):
    if pdtype == "si8": pdtype = "i8"
    if pdtype == "si16": pdtype = "i16"
    if pdtype == "si32": pdtype = "i32"
    if pdtype == "ui8": pdtype = "u8"
    if pdtype == "ui16": pdtype = "u16"
    if pdtype == "ui32": pdtype = "u32"
    if equal_dtypes.get(pdtype, pdtype) == input.dtype.name:
        res = input.reshape(pshape)
    elif pdtype == "i8" and input.dtype == np.float32:
        data = round_away_from_zero(input * pscale + pzero_point)
        res = np.clip(data, -128, 127).astype(np.int8).reshape(pshape)
    elif pdtype == "i8" and input.dtype == np.int32:
        data = round_away_from_zero(input * pscale + pzero_point)
        res = np.clip(data, -128, 127).astype(np.int8).reshape(pshape)
    elif pdtype == "u8" and input.dtype == np.int32:
        data = round_away_from_zero(input * pscale + pzero_point)
        res = np.clip(data, 0, 255).astype(np.int8).reshape(pshape)
    elif pdtype == "u8" and input.dtype == np.float32:
        data = round_away_from_zero(input * pscale + pzero_point)
        res = np.clip(data, 0, 255).astype(np.uint8).reshape(pshape)
    elif pdtype == "u16" and (input.dtype == np.float32 or input.dtype == np.int32):
        res = input.astype(np.uint16).reshape(pshape)
    elif pdtype == "i16" and (input.dtype == np.float32 or input.dtype == np.int32):
        res = input.astype(np.int16).reshape(pshape)
    elif pdtype == "f16" and input.dtype == np.float32:
        res = input.astype(np.float16)
    elif pdtype == "bf16" and input.dtype == np.float32:
        res = fp32_to_bf16(input).reshape(pshape)
    elif pdtype == "i32" and (input.dtype == np.float32 or input.dtype == np.int64):
        res = input.astype(np.int32).reshape(pshape)
    elif pdtype == "u32" and (input.dtype == np.float32 or input.dtype == np.int64 or input.dtype == np.uint32):
        res = input.astype(np.uint32).reshape(pshape)
    elif pdtype == "i4" and input.dtype == np.float32:
        data = round_away_from_zero(input * pscale + pzero_point)
        res = np.clip(data, -8, 7).astype(np.int8).reshape(pshape)
    elif pdtype == "u4" and input.dtype == np.float32:
        data = round_away_from_zero(input * pscale + pzero_point)
        res = np.clip(data, 0, 15).astype(np.uint8).reshape(pshape)
    elif pdtype == "f32":
        res = input.astype(np.float32).reshape(pshape)
    else:
        raise ValueError(f"unknown type: form {input.dtype} to {pdtype}")
    return res


def round_away_from_zero(x):
    a = np.floor(np.abs(x) + 0.5)
    return np.sign(x) * a


def bf16_to_fp32(d_bf16):
    s = d_bf16.shape
    d_bf16 = d_bf16.flatten()
    assert d_bf16.dtype == np.uint16
    d_fp32 = np.empty_like(d_bf16, dtype=np.float32)
    for i in range(len(d_bf16)):
        d_fp32[i] = struct.unpack("<f", struct.pack("<HH", 0, d_bf16[i]))[0]
    return d_fp32.reshape(s)


def fp32_to_bf16(d_fp32):
    s = d_fp32.shape
    d_fp32 = d_fp32.flatten()
    assert d_fp32.dtype == np.float32
    d_bf16 = np.empty_like(d_fp32, dtype=np.uint16)
    for i in range(len(d_bf16)):
        bytes = struct.pack("f", d_fp32[i])
        d_bf16[i] = struct.unpack("<H", struct.pack("BB", bytes[2], bytes[3]))[0]
    return d_bf16.reshape(s)
