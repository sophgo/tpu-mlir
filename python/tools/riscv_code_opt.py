#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2024 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import re
import argparse
import os
import shutil

def split_code_blocks(code):
    pattern = re.compile(r'^\s*//\s*"?([^"\n]+_gen_cmd)"?\s*$', re.MULTILINE)
    matches = list(pattern.finditer(code))
    if not matches:
        return [code]
    blocks = []
    start = 0
    for match in matches:
        if start < match.start():
            blocks.append(code[start:match.start()])
        end = matches[matches.index(match) + 1].start() if matches.index(match) + 1 < len(matches) else len(code)
        blocks.append(code[match.start():end])
        start = end
    return blocks


def extract_asm_name(asm_line):
    match = re.search(r'asm\s+volatile\s*\(\s*"([^"]+)"', asm_line)
    if match:
        return match.group(1).split()[0]
    return None

def extract_val_assignments(config):
    val_assignments = []
    for line in config:
        if 'val =' in line:
            val_assignments.append(line.strip())
    return val_assignments

def can_merge(last_asm1, last_asm2, config1, config2):
    # 提取汇编指令的名称
    asm_name1 = extract_asm_name(last_asm1)
    asm_name2 = extract_asm_name(last_asm2)
    if asm_name1 != asm_name2:
        return False

    val_assignments1 = extract_val_assignments(config1)
    val_assignments2 = extract_val_assignments(config2)
    if val_assignments1 != val_assignments2:
        return False

    return True

def generate_loop_code(loop_blocks):
    loop_code = loop_blocks[0].strip().split('\n')[:-1]
    loop_asm = loop_blocks[0].strip().split('\n')[-1]

    loop_code.append(f'    for (int i = 0; i < {len(loop_blocks)}; ++i) {{')
    loop_code.append(f'    {loop_asm}')
    loop_code.append('    }')

    return '\n'.join(loop_code) + '\n'

def identify_loops(blocks):
    optimized_blocks = []
    i = 0

    while i < len(blocks):
        current_block = blocks[i]
        current_block_lines = current_block.strip().split('\n')
        current_last_asm = current_block_lines[-1]

        loop_blocks = [current_block]
        j = i + 1

        while j < len(blocks):
            next_block = blocks[j]
            next_block_lines = next_block.strip().split('\n')
            next_last_asm = next_block_lines[-1]

            if can_merge(current_last_asm, next_last_asm, current_block_lines[:-1], next_block_lines[:-1]):
                loop_blocks.append(next_block)
                j += 1
            else:
                break

        if len(loop_blocks) > 1:
            loop_code = generate_loop_code(loop_blocks)
            optimized_blocks.append(loop_code)
            i = j
        else:
            optimized_blocks.append(current_block)
            i += 1

    return optimized_blocks

def generate_code(optimized_blocks):
    return '\n'.join(optimized_blocks)

def extract_last_asm_instruction(code_block):
    asm_pattern = re.compile(r'asm\s+volatile\s*\(\s*"([^"]+)"')
    matches = asm_pattern.findall(code_block)
    if matches:
        last_instruction = matches[-1].split()[0]
        return last_instruction
    return None

def remanage_blocks(blocks):
    return_blocks = []
    for block in blocks:
        return_blocks.append([extract_last_asm_instruction(block), block])
    return return_blocks

def remove_pure_gdma(index_arr, arr):
    new_index_arr = []
    start_set = set()
    for index in index_arr:
        if index[1] <= 1:
            continue
        if index[0] in start_set:
            continue
        count = 0
        for i in range(index[0], index[0] + index[1]):
            if arr[i].startswith('sg.dma'):
                count += 1
        if count != index[1]:
            start_set.add(index[0])
            new_index_arr.append(index)
    return new_index_arr

def find_repeated_subarrays(arr):
    n = len(arr)
    result = []

    # check overlap
    def is_overlapping(new_start, new_length, new_count):
        new_end = new_start + new_length * new_count
        for start, length, count in result:
            end = start + length * count
            if not (new_end <= start or new_start >= end):
                return True
        return False

    # find all possible for-loop
    i = 0
    while i < n:
        max_length = 0
        max_count = 0
        max_start = i

        # try different length
        for length in range(1, n - i + 1):
            count = 1
            while i + count * length < n and arr[i:i+length] == arr[i+length*count:i+length*(count+1)]:
                count += 1
            if count > 1 and length * count > max_length * max_count:
                if not is_overlapping(i, length, count):
                    max_length = length
                    max_count = count
                    max_start = i

        # record
        if max_length > 0 and max_count > 1:
            result.append((max_start, max_length, max_count))
            i = max_start + max_length * max_count - 1  # 跳过当前记录的子数组

        i += 1
    return remove_pure_gdma(result, arr)

def seize_addr(group_blocks, size, freq_, id_seq):
    group_blocks = [block[1] for block in group_blocks]
    group_blocks = [''.join(group_blocks[i:i + size]) for i in range(0, len(group_blocks), size)]
    freq = len(group_blocks)
    assert freq == freq_
    pattern = r"val = (0x[0-9a-fA-F]+);"
    replacement = "val = addr_{}_{}[i];".format(id_seq, '{}')
    values_length = len(re.findall(pattern, group_blocks[0]))
    template = group_blocks[0]
    for idx in range(values_length):
        replacement_ = replacement.format(idx)
        template = re.sub(pattern, replacement_, template, count=1)
    values = [[] for _ in range(values_length)]
    for block in group_blocks:
        for idx, match in enumerate(re.finditer(pattern, block)):
            if idx >= values_length:
                return None
            values[idx].append(match.group(1))
    values_conf_str = ''
    for value in values:
        if len(value) != freq:
            return None

    for i in range(values_length):
        values_conf_str += f'  const uint64_t addr_{id_seq}_{i}[{freq}] = {{'
        for j in range(freq):
            values_conf_str += f'{values[i][j]}, '
        values_conf_str = values_conf_str[:-2] + '};\n'

    values_conf_str += '\n  for (int i = 0; i < {}; ++i) {{'.format(freq)
    values_conf_str += '\n' + template + '\n  }'
    return values_conf_str

def replace_code_blocks(pure_code_block, index_arr, forloop_codes):
    # create a new code block list
    updated_code_block = []
    last_end = 0

    for i, (pos, size, rep_time) in enumerate(index_arr):
        # calculate start and end positions
        start = pos
        end = pos + size * rep_time

        # add un-replace block
        updated_code_block.extend(pure_code_block[last_end:start])

        # add replace block
        updated_code_block.append(forloop_codes[i])

        # update idx
        last_end = end

    # add rest codes
    updated_code_block.extend(pure_code_block[last_end:])

    return updated_code_block

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimize code blocks.')
    parser.add_argument('-f', '--file', type=str, help='Specify a C file')
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f'File {args.file} does not exist.')
        exit(-1)

    code = open(args.file).read()
    blocks = split_code_blocks(code)
    if len(blocks) == 1:
        exit(0)
    first_block = blocks[0]
    last_block = blocks[-1]
    code_blocks = remanage_blocks(blocks[1:-1])
    asm_names = [block[0] for block in code_blocks]
    # print(asm_names)
    index_arr = find_repeated_subarrays(asm_names)

    pure_code_block = [block[1] for block in code_blocks]
    forloop_codes = []
    group_block_test = [code_blocks[index[0]:index[0] + index[1] * index[2]] for index in index_arr]
    optimized_blocks = []
    n_index_arr = []
    for id, index in enumerate(index_arr):
        # print(index)
        # print(asm_names[index[0]:index[0] + index[1]])
        forloop_code = seize_addr(group_block_test[id], index[1], index[2], id)
        if forloop_code is not None:
            n_index_arr.append(index)
            forloop_codes.append(forloop_code)
    optimized_blocks = replace_code_blocks(pure_code_block, n_index_arr, forloop_codes)
    optimized_blocks = [first_block] + optimized_blocks + [last_block]
    # optimized_blocks = identify_loops(blocks)
    optimized_code = generate_code(optimized_blocks)
    file_orig = args.file[:-2] + '_orig' + args.file[-2:]
    shutil.copy(args.file, file_orig)
    with open(args.file, 'w') as f:
        f.write(optimized_code)
