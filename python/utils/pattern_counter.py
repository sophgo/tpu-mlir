#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import re
import argparse

class PatternCounter():
    def __init__(self, log_file):
        self.success_counter = {}
        self.log_file = log_file

    def count_matched_patterns(self):
        last_pattern = None
        pattern_name_regex = re.compile(r'Trying to match "(.+?)"')

        with open(self.log_file, 'r') as file:
            for line in file:
                match = pattern_name_regex.search(line)
                if match:
                    last_pattern = re.split(r'::(?![^<]*>)', match.group(1))[-1]
                elif '-> success' in line and last_pattern:
                    if last_pattern not in self.success_counter:
                        self.success_counter[last_pattern] = 1
                    else:
                        self.success_counter[last_pattern] += 1
                    last_pattern = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # yapf: disable
    parser.add_argument("--log_file", type=str, default=list(),
                        help="file that stores mlir debug info")
    # yapf: enable
    args, unknown_args = parser.parse_known_args()
    matcher = PatternCounter(args.log_file)
    matcher.count_matched_patterns()
    print(matcher.success_counter)



