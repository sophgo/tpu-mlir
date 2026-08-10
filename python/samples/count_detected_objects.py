#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
"""Customer example plugin: total number of detected objects (tier-3 demo).

Concrete adaptation of python/samples/custom_eval_template.py (the template
itself is NOT modified). It shows the FULL customer flow:

  a. PREPARE THE MODEL per trial -- ``begin_model`` runs ``model_deploy.py``
     at the beginning of every eval to compile the trial's quantized model to
     a bmodel (the place to also scp it to a board / open a remote session).
  b/c. PREPROCESS + INFERENCE + POSTPROCESS -- ``eval_sample(sample,
     preprocess)`` is the all-in-one per-sample hook: the customer decides
     where these run (locally, or on the board prepared in begin_model) and
     returns the PREDICTIONS. It is FAKED here (no real detector): draw a
     random per-sample object count whose upper bound scales with the float-
     layer count, so the score responds to the model under test.
  d. ACCURACY -- ``update`` runs on the HOST: fold each sample's predictions
     into the metric (a real detector compares vs the GT saved in init(); this
     demo has no GT and just sums the counts). Metric = SUM of per-sample
     counts; good enough when it reaches expected_acc (e.g. 500 over 100).

Config (python/samples/eval_config_count_objects.json):

    "eval_metric":  "custom"
    "score_plugin": "./count_detected_objects.py::count_detected_objects"
    "expected_acc": 500
    "sample_num":   100
    "direction":    "higher_better"
    "dataset":      "<imagenet val folder>"      # per-class subdirs (ImageFolder)
    "score_plugin_args": { "seed": 0, "base_ub": 8, "max_ub": 10, "deploy": true }

Real search wiring (unchanged):

    run_calibration.py model.mlir --search search_qtable \\
        --eval_config eval_config_count_objects.json \\
        --dataset /path/to/calib/data \\
        --calibration_table cali.txt --quantize_table out.qtable

Note: ``begin_model`` fires only at the search's count-decision points
(int8-baseline / initial-mix / adjust-*), not on the per-op sweep (that stays
on cos), so per-trial ``model_deploy`` runs only a handful of times.

Self-test (no model, deploy off): python3 count_detected_objects.py
"""
import os
import subprocess

import numpy as np

# Dual import so this file works from anywhere (see custom_eval_template.py).
try:
    from eval.postprocess_and_score_calc.base_class import base_class
except ImportError:  # pragma: no cover - future public alias
    from tpu_mlir.eval_plugin import base_class


class count_detected_objects(base_class):
    """File stem == class name, so the config may omit the '::ClassName' part.

    data_mode stays 'toolkit': the harness owns dataset selection + the mlir's
    ImageNet preprocessing + simulator inference. This plugin only adds a
    per-trial model-prepare step (begin_model) and a faked metric.
    """

    data_mode = 'toolkit'

    # Set to False (or score_plugin_args {"debug_hooks": false}) to silence the
    # per-interface call tracing added below.
    def _dbg(self, msg):
        """Print a hook-call trace line so the exact interface call sequence is
        visible in the search log. Prefix is greppable: 'HOOK'."""
        if getattr(self, '_debug_hooks', True):
            print('[count_detected_objects][HOOK] {}'.format(msg))

    def init(self, args):
        """Parse config once per search; survives across trials."""
        self.args = args
        # Fake-metric knobs (flat attributes from score_plugin_args).
        self.seed = int(getattr(args, 'seed', 0))
        self.base_ub = int(getattr(args, 'base_ub', 8))  # int8-baseline upper bound
        self.max_ub = int(getattr(args, 'max_ub', 10))  # cap as float layers grow
        # Whether begin_model actually runs model_deploy (turn off for a quick
        # smoke test where no bmodel toolchain / dataset is available).
        self.deploy = bool(getattr(args, 'deploy', True))
        self.deploy_dir = getattr(args, 'deploy_dir', '.') or '.'
        self._debug_hooks = bool(getattr(args, 'debug_hooks', True))
        # Per-trial state (cleared by reset()).
        self.idx = 0
        self.total = 0
        self.n_fp = 0  # float-layer count of the model under test (set per trial)
        self._eval_calls = 0  # per-trial eval_sample() call counter (for tracing)
        # Deterministic RNG: reseeded every trial so all trials see the SAME
        # random draws -- only the upper bound (n_fp) differs between models.
        self._rng = np.random.default_rng(self.seed)
        self._dbg('init(args) called once per search: seed={} base_ub={} max_ub={} '
                  'deploy={}'.format(self.seed, self.base_ub, self.max_ub, self.deploy))

    def reset(self):
        """Clear per-trial counters and reseed the RNG (same draws each trial);
        config parsed in init() is kept."""
        self._dbg('reset() called (start of a trial): clearing idx/total, reseeding RNG')
        self.idx = 0
        self.total = 0
        self._eval_calls = 0
        self._rng = np.random.default_rng(self.seed)

    # --- a. prepare the model at the beginning of every eval --------------
    def begin_model(self, model_ctx):
        """Called once per trial, BEFORE the sample loop. Deploy the trial's
        quantized model to a bmodel (and, in a real HIL setup, download it to
        the board / open a remote session here).

        Returns None: the harness still drives the ImageNet preprocess +
        inference loop (this plugin only adds the deploy step + fakes the
        metric; it does not take over the loop -- that would be 'self')."""
        self._dbg('begin_model(model_ctx) called: phase={} chip={} mix_table={} '
                  'quanted_mlir_file={}'.format(getattr(model_ctx, 'phase', None),
                                                getattr(model_ctx, 'chip', None),
                                                getattr(model_ctx, 'mix_table', None),
                                                getattr(model_ctx, 'quanted_mlir_file', None)))
        # Learn WHICH model this trial is: the float-layer count sets the fake
        # accuracy's upper bound (more float layers -> higher count).
        self.n_fp = self._count_float_layers(model_ctx)
        self._dbg('begin_model: resolved n_fp={} -> fake upper bound ub={}'.format(
            self.n_fp, min(self.max_ub, self.base_ub + self.n_fp)))
        self._model_deploy(model_ctx)
        self._dbg('begin_model returning None (harness will drive samples + inference)')
        return None

    def end_model(self):
        """Per-trial teardown -- in a real HIL setup, close the board session
        / free the deployed bmodel here."""
        self._dbg('end_model() called (end of trial teardown)')

    def _count_float_layers(self, model_ctx):
        """Float-layer count of the model under test. Prefer the harness-
        provided ``n_fp``; else count the ops listed in the trial mix table."""
        n_fp = getattr(model_ctx, 'n_fp', None)
        if isinstance(n_fp, int):
            return n_fp
        mix_table = getattr(model_ctx, 'mix_table', None)
        if mix_table and os.path.isfile(mix_table):
            n = 0
            for line in open(mix_table):
                s = line.strip()
                if s and not s.startswith('#'):
                    n += 1
            return n
        return 0

    def _model_deploy(self, model_ctx):
        """Run model_deploy.py on the trial model to build a bmodel. Failures
        are logged, not raised -- the faked metric does not depend on the
        bmodel, and a real customer would gate the search on their own remote
        result instead."""
        if not self.deploy:
            print('[count_detected_objects] deploy disabled (score_plugin_args.deploy=false); '
                  'phase={} n_fp={}'.format(getattr(model_ctx, 'phase', '?'), self.n_fp))
            return
        mlir = getattr(model_ctx, 'fp32_mlir', None)
        chip = getattr(model_ctx, 'chip', None)
        cali = getattr(model_ctx, 'cali_table', None)
        mix_table = getattr(model_ctx, 'mix_table', None)
        if not mlir or not chip:
            print('[count_detected_objects] skip model_deploy: missing mlir/chip in model_ctx')
            return
        phase = str(getattr(model_ctx, 'phase', 'trial')).replace('/', '_')
        bmodel = os.path.join(self.deploy_dir, 'trial_{}.bmodel'.format(phase))
        cmd = ['model_deploy.py', '--mlir', mlir, '--quantize', 'INT8', '--processor', chip]
        if cali:
            cmd += ['--calibration_table', cali]
        if mix_table and os.path.isfile(mix_table):
            cmd += ['--quantize_table', mix_table]
        cmd += ['--model', bmodel]
        # TODO(customer): after a successful deploy, scp the bmodel to the
        # board and start your remote eval session here.
        print('[count_detected_objects] model_deploy (phase={} n_fp={}): {}'.format(
            phase, self.n_fp, ' '.join(cmd)))
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            print('[count_detected_objects] deployed -> {}'.format(bmodel))
        except Exception as e:  # noqa: BLE001 - demo must not crash the search
            print('[count_detected_objects] model_deploy failed (continuing): {}'.format(e))

    # --- b/c. eval_sample: preprocess + inference + postprocess per sample -
    def preproc(self, img_paths):
        return None  # unused: eval_sample owns preprocessing (required by ABC)

    def eval_sample(self, sample, preprocess):
        """All-in-one per-sample hook: return the PREDICTIONS for one sample.

        A real HIL plugin would:
            x = preprocess(sample)          # reuse the common ImageNet pre
            out = self.session.infer(x)     # run on the board (from begin_model)
            return decode_boxes(out)        # postprocess -> detections
        Here it is FAKED: draw a random per-sample object count whose upper
        bound grows with the model's float-layer count (n_fp), so the score
        responds to the model under test. The returned count is the compact
        "predictions" that cross back to the host; update() accumulates it.
        """
        self._eval_calls += 1
        if self._eval_calls == 1 or self._eval_calls % 25 == 0:
            self._dbg('eval_sample(sample, preprocess) call #{} '
                      '(preprocess+infer+postprocess; here faked)'.format(self._eval_calls))
        # TODO(customer): x = preprocess(sample); out = board.infer(x);
        #                 return decode_and_count(out)
        ub = min(self.max_ub, self.base_ub + self.n_fp)
        return int(self._rng.integers(0, ub + 1))

    # --- accuracy: host-side accumulate of the returned predictions -------
    def update(self, idx, predictions, sample=None, labels=None, ratios=None):
        """HOST side: fold one sample's predictions into the metric. For a real
        detector this is where predictions meet the GT saved in init(); this
        demo has no GT, so it just sums the per-sample counts."""
        self.idx = idx + 1
        self.total += int(predictions)

    def get_result(self):
        """Metric = TOTAL object count over the eval set (never None).
        Empty/degraded trial -> 0.0 (worst score for higher_better)."""
        self._dbg('get_result() called after {} eval_sample calls -> total={}'.format(
            getattr(self, '_eval_calls', 0), self.total))
        return float(self.total)

    def print_info(self):
        self._dbg('print_info() called')
        if self.idx > 0:
            print('[{}] idx:{}, total_objects:{}, avg/sample:{:.2f}, n_fp:{}'.format(
                self.__class__.__name__, self.idx, self.total, self.total / self.idx, self.n_fp))


if __name__ == '__main__':
    # Self-test (deploy off, no model): the faked score must rise with n_fp.
    import types

    def _run(n_fp, samples=100):
        p = count_detected_objects(types.SimpleNamespace(seed=0, deploy=False))
        # Search order per trial: reset() -> begin_model() -> eval_sample()*N.
        p.reset()
        p.begin_model(types.SimpleNamespace(chip=None, fp32_mlir=None, n_fp=n_fp, phase='t'))
        for i in range(samples):
            preds = p.eval_sample('img_{}.jpg'.format(i),
                                  lambda s: None)  # faked; preprocess unused
            p.update(i, preds)
        p.print_info()
        p.end_model()
        return p.get_result()

    base = _run(0)  # int8 baseline: ub=8 -> ~400
    more = _run(2)  # +float layers: ub=10 -> ~500
    assert 0.0 < base < more, (base, more)
    assert _run(0) == base  # deterministic across runs (seeded)
    print('self-test OK: total(n_fp=0)={:.0f} < total(n_fp=2)={:.0f}, deterministic'.format(
        base, more))
