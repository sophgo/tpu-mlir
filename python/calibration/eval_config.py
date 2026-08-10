#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
"""Config object for end-to-end / custom eval in the search loop.

All end-to-end-related config (dataset, label file, postprocess plugin,
sample count, expected accuracy, direction, custom func path) lives here,
loaded from a JSON file via ``--eval_config <path>`` on ``run_calibration.py``.
This keeps the calibration CLI focused on cos-based search params; end-to-end
config is isolated in JSON.

Default (no ``--eval_config``): ``EvalConfig()`` -> ``eval_metric='cos'``,
the current behavior. End-to-end path is purely additive.
"""
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvalConfig:
    """End-to-end eval config; ``eval_metric='cos'`` is the default that
    preserves current search behavior.

    Fields used only by end-to-end / custom paths are ignored when
    ``eval_metric == 'cos'``.
    """

    # --- the metric option set ---
    # eval_metric is the single source of truth for the decision metric. It
    # partitions into two families:
    #   COS_METRICS     = {'cos'}           -> threshold is expected_cos (> compare)
    #   END2END_METRICS = {'topx',          -> threshold is expected_acc (direction compare);
    #                       'coco_mAP',         differ only by score plugin + metric_index:
    #                       'custom'}           topx -> top1/top5, coco_mAP -> mAP/mAP50, custom -> user
    # All end2end metrics share expected_acc + direction + the same end2end
    # eval path; the plugin/metric_index selects the concrete number. This is
    # why they bind together as one family.
    COS_METRICS = ('cos', )
    END2END_METRICS = ('topx', 'coco_mAP', 'custom')

    # --- core ---
    eval_metric: str = 'cos'  # cos | topx | coco_mAP | custom
    direction: str = 'higher_better'  # 'higher_better' | 'lower_better'
    expected_acc: float = 0.0  # end2end floor (top1/mAP/custom); replaces expected_cos
    metric_index: int = 0  # which tuple element of get_result() to use (topx: 0=top1, 1=top5)
    # Scope knob for the EXPENSIVE per-op sensitivity sweep only. The cheap
    # float-layer COUNT decision points (early-exit, adjust_qtable binary
    # search, fast-search round loop) always use the configured eval_metric,
    # so expected_acc drives float-layer selection whenever eval_metric is an
    # end2end metric -- regardless of this flag. When true, the per-op sweep
    # ALSO runs on the end2end metric (slow: sample_num imgs x #ops); when
    # false (default), the sweep stays on cheap cos. Ignored for eval_metric='cos'.
    use_end2end_in_search: bool = False

    # --- end-to-end / custom (None/empty when eval_metric == 'cos') ---
    score_plugin: Optional[
        str] = None  # 'topx' | 'coco_mAP' | '<custom>' (base_class subclass name)
    score_plugin_args: Dict[str, Any] = field(default_factory=dict)
    input_source: str = 'calibration'  # 'calibration' (reuse ref_activations) | 'full' (re-preprocess)
    sample_num: int = 0  # 0 = all
    dataset: Optional[str] = None  # only for input_source='full'
    data_list: Optional[str] = None  # only for input_source='full'
    postprocess_type: Optional[str] = None  # mirrors model_eval.py's --postprocess_type

    # --- custom path only ---
    eval_func: Optional[str] = None  # 'path/to/file.py::func_name'
    gt_provider: Optional[str] = None  # optional 'path/to/file.py::func_name'

    @classmethod
    def from_json(cls, path: Optional[str]) -> 'EvalConfig':
        if not path:
            # Default: load cos_eval.json shipped alongside this module.
            # Equivalent to EvalConfig() but surfaces the documented template
            # (the file lives at python/calibration/cos_eval.json).
            default_path = os.path.join(os.path.dirname(__file__), 'cos_eval.json')
            if os.path.exists(default_path):
                path = default_path
            else:
                cfg = cls()
                cfg._config_dir = os.getcwd()
                return cfg
        with open(path) as f:
            data = json.load(f)
        # Allow shorthand: score_plugin defaults to eval_metric when not set
        if 'score_plugin' not in data and data.get('eval_metric') in ('topx', 'coco_mAP'):
            data['score_plugin'] = data['eval_metric']
        if 'postprocess_type' not in data and data.get('score_plugin'):
            data['postprocess_type'] = data['score_plugin']
        # Filter to known fields; ignore unknown keys (forward-compat).
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        cfg = cls(**filtered)
        # Directory of the JSON file — external score_plugin paths
        # ('./my_metric.py::MyMetric') resolve against it (then CWD).
        cfg._config_dir = os.path.dirname(os.path.abspath(path))
        cfg._validate()
        return cfg

    def _validate(self) -> None:
        """Validate the eval config to catch silent misconfigurations.

        - eval_metric must be a known member of the option set.
        - expected_acc must be > 0 for end-to-end metrics (it is the float-layer
          selection floor / early-exit target used at every count-decision
          point; if 0 the comparison against a top1/mAP score ~0.85 would be
          meaningless and nothing gets demoted).
        - sample_num must be > 0 for end-to-end (0 means "all images", which
          for a full val set is tens of thousands of inferences PER trial —
          almost never intended; force an explicit count).
        """
        valid = self.COS_METRICS + self.END2END_METRICS
        if self.eval_metric not in valid:
            raise ValueError("eval_config: eval_metric must be one of {}. Got '{}'.".format(
                valid, self.eval_metric))
        if not self.is_end_to_end:
            return
        if self.expected_acc <= 0:
            raise ValueError("eval_config: end-to-end metric '{}' requires expected_acc > 0 "
                             "(the minimum acceptable {} used as the float-layer selection "
                             "floor / early-exit target). Got expected_acc={}.".format(
                                 self.eval_metric, self.eval_metric, self.expected_acc))
        if self.sample_num <= 0:
            raise ValueError("eval_config: end-to-end metric '{}' requires sample_num > 0 "
                             "(images per trial). sample_num=0 means the whole dataset, "
                             "which is tens of thousands of inferences per trial. Set an "
                             "explicit count (e.g. 50-1000).".format(self.eval_metric))
        if self.eval_metric == 'custom' and not self.score_plugin:
            raise ValueError("eval_config: eval_metric 'custom' requires 'score_plugin' "
                             "(a built-in name or 'path/to/file.py::ClassName', e.g. "
                             "'./my_metric.py::MyMetric')")

    @property
    def is_end_to_end(self) -> bool:
        return self.eval_metric in self.END2END_METRICS

    @property
    def use_end2end_everywhere(self) -> bool:
        """True when the EXPENSIVE per-op sensitivity sweep should also run on
        the end2end metric (not just the cheap count-decision points, which
        always use the configured metric). Requires an end2end eval_metric +
        use_end2end_in_search. Used only to scope the per-op sweep."""
        return self.is_end_to_end and self.use_end2end_in_search
