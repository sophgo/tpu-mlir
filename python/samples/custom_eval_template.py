#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
"""Custom eval plugin template -- copy anywhere, edit, and plug into the
search_qtable end-to-end eval path via --eval_config only (no source access,
no rebuild). See docs/custom-end2end-eval-plugin-proposal.md.

Workflow:

  1. Copy this file next to your eval config (any directory works):
       cp python/samples/custom_eval_template.py  ./my_metric.py
     Rename the class to your metric (or keep it -- see step 3).

  2. Edit every ``# TODO(customer):`` marker below. The default dummy metric
     (running mean of the max output value) runs out-of-the-box so you can
     smoke-test the wiring before implementing your real metric.

  3. Copy python/samples/eval_config_custom.json next to your copied file and
     point the plugin fields at it:
         "score_plugin":     "./my_metric.py::MyMetric"
         "postprocess_type": "./my_metric.py::MyMetric"
     The class name after ``::`` defaults to the file stem when omitted.
     Relative paths resolve against the eval-config JSON directory, then CWD.

  4. Run the search with the config only (everything else unchanged):
       run_calibration.py model.mlir --search search_qtable \\
           --eval_config my_config.json --dataset /path/to/calib/data \\
           --calibration_table cali.txt --quantize_table out.qtable

  Optional pre-check outside the search (strongly recommended before a
  multi-hour search): python/samples/end2end_custom_example.py.

Contract (subclass of base_class; the search calls each method):

  init(args)          once per search     parse GT/config once, keep on self
  reset()             between trials      clear per-trial counters, KEEP GT
  preproc(img_paths)  optional            return None -> use the mlir img_proc
  update(...)         per image/batch     accumulate model outputs vs GT
  get_result()        once per trial      float or tuple of floats, NEVER None
  print_info()        progress logging    optional

Result rules that matter most:
  - get_result() must return a float or a tuple/list of floats -- never None.
    metric_index in the JSON picks the decision element.
  - direction in the JSON: 'higher_better' -> score >= expected_acc is good;
    'lower_better' -> score <= expected_acc (perplexity/WER/error-rate).
  - On a degraded trial, return the worst score for your direction (e.g. 0.0
    for higher_better) so the trial fails cleanly instead of crashing the
    search. (Historical lesson: a plugin returning None crashed the search.)

args plumbing: entries of score_plugin_args from the JSON (except
label_file/batch_size) arrive as flat attributes on args; label_file arrives
as args.label_file.

Self-test (no model needed): python custom_eval_template.py

A worked example built from this template (uses the tier-3 begin_model hook to
run model_deploy per trial, keeps the common ImageNet preprocess, and fakes a
random metric): python/samples/count_detected_objects.py.
"""
import os

import numpy as np

# Dual import so the copied file works from anywhere: in-tree layout first
# (dev container / source install), then the future public wheel alias.
# NOTE: do not use a relative import (from .base_class import *) -- this file
# is loaded by path (importlib), which has no package context.
try:
    from eval.postprocess_and_score_calc.base_class import base_class
except ImportError:  # pragma: no cover - future public alias
    from tpu_mlir.eval_plugin import base_class


class custom_eval_template(base_class):
    """TODO(customer): rename this class to your metric name.

    Keeping file name == class name lets the config omit the '::ClassName'
    part; otherwise always spell it out: './my_metric.py::MyMetric'.
    """

    def init(self, args):
        """Parse config/GT once per search. Called by base_class.__init__.

        Keep everything you parse here on ``self`` -- it survives across
        trials (reset() must NOT clear it).
        """
        self.args = args
        # Per-trial accumulators (cleared by reset()).
        self.idx = 0
        self._sum = 0.0
        # TODO(customer): replace the dummy accumulator above with your own,
        # e.g. self._hits = 0, self._total_loss = 0.0, ...

        # Label plumbing compatible with MixModelInference's fallback sample
        # list: expose all_labels_dict / all_labels_list parsed from
        # args.label_file (same format as model_eval.py / topx.py).
        self.all_labels_dict = {}
        self.all_labels_list = []
        label_file = getattr(args, 'label_file', '')
        if label_file and os.path.exists(label_file):
            for line in open(label_file, 'r').readlines():
                parts = [i for i in line.strip().split(' ') if len(i.strip()) > 0]
                if len(parts) == 2:
                    self.all_labels_dict[parts[0]] = int(parts[1])
                else:
                    self.all_labels_list.append(int(parts[0]))

        # TODO(customer): parse your own ground truth here once (json/csv/
        # database), e.g. self.gt = json.load(open(args.gt_json)).

    def reset(self):
        """Clear per-trial accumulators; KEEP the GT parsed in init().

        Called by MixModelInference.reset_score() between trials -- the model
        changes per trial, the GT does not. Forgetting this makes every trial
        accumulate on top of the previous one (silently wrong scores).
        """
        self.idx = 0
        self._sum = 0.0
        # TODO(customer): clear your accumulators here too.

    def preproc(self, img_paths):
        """Optional custom preprocessing. Return None to use the mlir's
        img_proc (recommended for classification-style inputs).

        Only called when the pipeline runs with debug_cmd
        'not_use_preprocess' (detection-style plugins do their own
        letterbox/resize here and fill ratio_list). Returning None is the
        right default for the minimal scope.
        """
        return None

    def update(self, idx, outputs, img_paths=None, labels=None, ratios=None):
        """Accumulate one sample (or batch).

        Arg 2 is either the raw model ``outputs`` (built-in path: harness ran
        img_proc + inference) OR your ``predictions`` (if you override
        ``eval_sample`` -- see below); it is whatever your own code produced,
        so treat it accordingly. This is the HOST side: compare against the GT
        you saved in ``init`` and accumulate."""
        self.idx = idx + 1
        # GT resolution example, same precedence as topx.py: explicit labels
        # arg > positional label list > filename-matched label dict.
        label = None
        if labels is not None:
            label = labels[0] if isinstance(labels, (list, tuple)) else labels
        elif len(self.all_labels_list) > 0:
            label = self.all_labels_list[idx]
        elif len(self.all_labels_dict) > 0 and img_paths is not None:
            for key in self.all_labels_dict:
                if img_paths.endswith(key):
                    label = self.all_labels_dict[key]
                    break

        out = np.asarray(outputs[0])
        # Dummy metric: running mean of the max output value. Runs out of the
        # box so the wiring can be smoke-tested before you implement yours.
        self._sum += float(np.max(out))
        # TODO(customer): replace the line above with your real accumulation,
        # comparing ``out`` (and ``label``) against your GT, e.g.:
        #   top1 = int(out.reshape(-1).argsort()[::-1][0])
        #   self._hits += int(top1 == label)

    def get_result(self):
        """Return the metric. MUST be a float or a tuple/list of floats --
        never None. metric_index in the JSON picks the decision element."""
        # Worst-possible score on an empty/degraded trial so it fails cleanly
        # (higher_better assumed; invert for lower_better metrics).
        mean = self._sum / self.idx if self.idx > 0 else 0.0
        # Returning a 2-tuple demonstrates metric_index; return a plain float
        # if your metric is a single number (then use metric_index 0).
        return (mean, mean)

    def print_info(self):
        """Optional progress logging (called periodically + before result)."""
        if self.idx > 0:
            print('[{}] idx:{}, running_mean_max:{:.4f}'.format(self.__class__.__name__, self.idx,
                                                                self._sum / self.idx))
        # TODO(customer): log your running metric here.

    # ======================================================================
    # OPTIONAL advanced hooks — uncomment/override only if you need them.
    # The 6 methods above are enough for the common case (harness supplies
    # the dataset, preprocess and inference). These hooks let you take over
    # more of the pipeline. All are no-ops by default (from base_class).
    # ======================================================================

    # --- tier-3: prepare / tear down the model per trial ------------------
    # begin_model runs once per trial, AFTER reset(), BEFORE the sample loop.
    # It receives model_ctx describing the model under test. Two patterns:
    #
    # (A) DEPLOY then return None -- keep the harness driving the common
    #     preprocess + inference; you just add a per-trial deploy step (and,
    #     if you want, fake / adjust the metric using model_ctx.n_fp). This is
    #     what python/samples/count_detected_objects.py does:
    #
    #   def begin_model(self, model_ctx):
    #       # model_ctx: chip, fp32_mlir, quanted_mlir_file, weight_file,
    #       #            cali_table, mix_table, n_fp, phase, using_cuda, ...
    #       # TODO(customer): compile model_ctx.fp32_mlir (+cali/mix table) to
    #       #   a bmodel via model_deploy.py; scp to board; open session.
    #       self.n_fp = getattr(model_ctx, 'n_fp', 0) or 0
    #       return None            # harness still drives samples + inference
    #
    # (B) return 'self' -- take over the WHOLE eval (own dataset + preprocess
    #     + inference, e.g. a remote board run); the harness skips its loop and
    #     only reads get_result(). Accumulate here freely (runs after reset()):
    #
    #   def begin_model(self, model_ctx):
    #       deploy_and_open_session(model_ctx)
    #       for sample in my_dataset:          # SAME order every trial
    #           out = remote_infer(sample)
    #           self.total += count_objects(out)
    #       return 'self'
    #
    # def end_model(self):
    #     pass    # TODO(customer): close the remote session / free handles

    # --- eval_sample: own preprocess+inference+postprocess per sample -----
    # The all-in-one per-sample hook. Override it to decide WHERE those three
    # run (locally, or ship the sample to the board prepared in begin_model)
    # and return the PREDICTIONS. GT stays on the host: your update() compares
    # predictions vs GT. Return None to fall back to img_proc + simulator.
    #
    # def eval_sample(self, sample, preprocess):
    #     # sample: harness img_path (tier-1) or your object (tier-2 __iter__)
    #     # preprocess: harness's common img_proc -- call it or do your own.
    #     x = preprocess(sample)                 # optional: reuse ImageNet pre
    #     out = self.session.infer(x)            # your board / runtime
    #     return decode_predictions(out)         # opaque; consumed by update()

    # --- tier-2: own dataset + enumeration (harness runs inference) -------
    # Set the class attribute data_mode = 'plugin' (top of the class) and
    # implement open_dataset/__iter__/preproc. The harness then enumerates
    # YOUR samples and calls YOUR preproc, but still runs the (simulator)
    # inference; your update() receives the raw sample as its 3rd arg.
    # (Combine with eval_sample to also own inference.)
    #
    # data_mode = 'plugin'
    #
    # def open_dataset(self, args):
    #     # TODO(customer): load your dataset once (index/annotations).
    #     self._samples = list(load_my_dataset(args.dataset))
    #
    # def __iter__(self):
    #     # MUST yield the SAME samples in the SAME order every trial.
    #     return iter(self._samples)
    #
    # def preproc(self, sample):
    #     # TODO(customer): sample -> model input tensor (or list of tensors).
    #     return my_preprocess(sample)


if __name__ == '__main__':
    # Self-test: exercise the contract without a model. A plugin that passes
    # here will load and run in the search loop (loader checks the same 6
    # methods + get_result() must never return None).
    import types

    args = types.SimpleNamespace(label_file='', batch_size=1)
    plugin = custom_eval_template(args)
    rng = np.random.default_rng(0)
    for i in range(4):
        fake_outputs = [rng.random((1, 10)).astype(np.float32)]
        plugin.update(i, fake_outputs, img_paths='img_{}.jpg'.format(i))
    result = plugin.get_result()
    assert isinstance(result, (tuple, list)) and len(result) > 0
    assert all(isinstance(v, float) for v in result)
    assert plugin.idx == 4
    plugin.reset()
    assert plugin.idx == 0 and plugin._sum == 0.0
    assert plugin.all_labels_dict == {}  # GT survives reset (empty here)
    print('self-test OK: get_result() = {}'.format(result))
