#!/usr/bin/python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import re
import os
import sys
import logging
from collections import namedtuple

def re_key_value(prefix, key_str:str):
    keys = key_str.split(" ")
    segs = [".*"+prefix+".*"]
    for key in keys[:-1]:
        if key == "":
            continue
        seg = r"{}=(?P<{}>\S+)".format(key, key)
        segs.append(seg)
    seg = r"{}=(?P<{}>.*)".format(keys[-1], keys[-1])
    segs.append(seg)
    return re.compile(r"\s*".join(segs))

def load_module(filename, name=None):
    if not filename.endswith(".py"):
        return None
    if name is None:
        name = os.path.basename(filename).split(".")[0]
    if not name.isidentifier():
        return None
    with open(filename, "r", encoding="utf8") as f:
        code = f.read()
        module = type(sys)(name)
        sys.modules[name] = module
        try:
            exec(code, module.__dict__)
            return module
        except (EnvironmentError, SyntaxError) as err:
            sys.modules.pop(name, None)
            print(err)
    return sys.modules.get(name, None)

def load_arch_lib(arch):
    archlib_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        arch.name + "_defs.py")
    return load_module(archlib_path)

def next_id_by_width(id_val, inc=1, width=16):
    new_id = id_val+inc
    while new_id>=(1<<width):
        new_id -= (1<<width)
    return new_id

def enum_cast(value, enum_type, default_val=-1):
    try:
        return enum_type(value)
    except:
        logging.warn("{} is not a valid {} value, using default({}) instead. ".format(value, enum_type.__name__, default_val))
        return enum_type(default_val)

def option_to_map(option:str):
    kv_strs = option.split(";")
    options={}
    for kv_str in kv_strs:
        kv_str = kv_str.strip()
        idx = kv_str.find('=')
        if idx<0:
            continue
        key = kv_str[0:idx]
        val = kv_str[idx+1:]
        options[key] = val
    return options

def usec_to_str(usec:int):
    if usec<1000:
        return "%dus"%(usec)
    elif usec<1000000:
        return "%.3fms"%(usec/1000.0)
    else:
        return "%.3fs"%(usec/1000000.0)

def print_table(title, header, data, is_pretty=False):
    width = [len(h) for h in header]
    for d in data:
        width[:len(d)] = [max(width[i], len(str(s))) for i,s in enumerate(d)]
    sep = " | " if is_pretty else "  "
    format_str = " " + sep.join(["{:>%d}"%w for w in width]) + " "
    total_width = sum(width) + (len(width)-1) * len(sep) + 2
    if is_pretty:
        print("\033[7;4;33m", end="")

    title_format="{:^%d}"%total_width
    print(title_format.format(title))

    if is_pretty:
        print("\033[0m", end="")

    if is_pretty:
        print("\033[1;4;33m", end="")
    print(format_str.format(*header))
    if is_pretty:
        print("\033[0m", end="")
    hlen = len(header)
    temp = [""]*hlen
    colors=["\033[1;4;36m", "\033[1;4;32m"]
    for d in data:
        if is_pretty:
            print(colors[0], end="")
            colors[0], colors[1] = colors[1], colors[0]
        sd = [str(s) for s in d]
        sd += temp[0:hlen-len(d)]
        print(format_str.format(*sd))
        if is_pretty:
            print("\033[0m", end="")

def filter_items(items, filter_expr, order_by, is_desc, max_size, columns = []):
    new_items = []
    for i in items:
        if type(i) != dict:
            i = i._asdict()
        if eval(filter_expr, i):
            new_items.append(i)
    def key_func(a):
        return [a[o] for o in order_by if a.get(o) is not None]
    if order_by is not None:
        new_items = sorted(new_items, key = key_func, reverse = is_desc)

    if len(columns)>0:
        tmp_items = []
        for i in new_items:
            tmp_items.append([i[c] if i.get(c) is not None else "" for c in columns])
        new_items = tmp_items
    else:
        new_items = [i.values() for i in new_items]

    if max_size>0 and max_size<len(new_items):
        new_items = new_items[0:max_size]
    return new_items

def calc_bandwidth(num_bytes, dur_usec):
    bandwidth = num_bytes/dur_usec*1e6
    if bandwidth>1e9:
        return "%.2fGB/s"%(bandwidth/1e9)
    elif bandwidth>1e6:
        return "%.2fMB/s"%(bandwidth/1e6)
    elif bandwidth>1e3:
        return "%.2fKB/s"%(bandwidth/1e3)
    return "%.2fB/s"%bandwidth

if __name__ == "__main__":
    print(usec_to_str(200))
    print(usec_to_str(234500))
    print(usec_to_str(234567800))
    options = option_to_map("name=xiaotan;status=ok=2; hello=world")
    print(options.get("status"))
    print(options.items())
    # print_table("hello", ["name", "age"], [["xiaotan" ], ["zhangsan", 5], ["lisi", 40]], True)
    Item = namedtuple("Item", "name age status info")
    data = [
        Item("zhang", 3, "ok", "student"),
        Item("Tan", 5, "ok", "baby"),
        Item("Li", 4, "error", "worker"),
        Item("Wang", 30, "ok", "man"),
        Item("Feng", 26, "ok", "girl"),
    ]
    new_data = filter_items(data, "age>=5", ["status", "age"], False, 3, ["name", "age", 'info'])
    print_table("Items", ["name","age", "info"], new_data, True)
