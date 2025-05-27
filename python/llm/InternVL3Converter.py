# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from .LlmConverter import *
from typing_extensions import override


class InternVL3Converter(LlmConverter):

    def __init__(self, args, config):
        super().__init__(args, config)

    @override
    def load_pretrained(self, config):
        super().load_pretrained(config)
        self.llm_config = config.llm_config
        self.llm_type = self.llm_config.model_type
