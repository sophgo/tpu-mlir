# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from .LlmConverter import *
from typing_extensions import override


class Gemma3Converter(LlmConverter):

    def __init__(self, args, config):
        super().__init__(args, config)

    @override
    def load_pretrained(self, config):
        super().load_pretrained(config)
        self.model_info = GEMMA3_INFO
        self.llm_config = config.text_config
        self.llm_type = LlmType.GEMMA3

    @override
    def init_config(self):
        super().init_config()
        self.tie_word_embeddings = True
        self.do_lmhead_merge = self.tie_word_embeddings and not self.embedding_disk and self.num_device < 2
