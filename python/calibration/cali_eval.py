#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
"""Search-loop eval axis + the bridge to ``model_eval.py``.

This module consolidates everything eval-related that's specific to the
search loop:

- ``EvalBase`` holds the build -> compute_score -> clean -> judge
  skeleton, plus ``target`` / ``direction`` instance state and a generic
  base ``_judge``.
- ``CosEval`` covers all cos-based eval modes via ctor configuration.
- ``End2EndEval`` (Phase 4) and ``CustomEval`` (Phase 6) are peer subclasses.
- ``MixModelInference`` bridges the in-memory ``MixQuantModel`` from the
  search loop into the ``model_eval.py`` scoring pipeline (subclass of
  ``common_inference``; reuses ``img_proc`` / ``score`` / ``model_invoke``
  / ``get_result``; overrides ``__init__`` / ``invoke``; adds
  ``set_model`` / ``reset_score`` / ``get_sample`` for the per-trial swap
  pattern).

Import-graph notes:
- This module imports the dataclasses (``Verdict``, ``Trial``,
  ``SearchState``) and ``EvalStrategy`` from ``search_loop`` at module
  level. ``search_loop.py`` must NOT import from this module at module
  level — eval classes are constructed in ``search_qtable.py``, which
  imports both. Lazy import in ``search_loop.py`` if any type ref is
  needed.
- ``MixModelInference`` imports ``common_inference`` from
  ``python/eval/model_inference.py`` at module level — the
  ``python/eval/`` subsystem is otherwise untouched.

See ``docs/search-qtable-eval-merge-supplement.html`` for the design.
"""
import collections
from typing import Any, Dict, List, Optional

from .search_loop import EvalStrategy, SearchState, Trial, Verdict

__all__ = [
    'EvalBase',
    'CosEval',
    'End2EndEval',
    'CustomEval',
    'EvalStrategy',
    'Verdict',
    'Trial',
    'SearchState',
    'MixModelInference',
    'resolve_score_plugin',
]


class EvalBase(EvalStrategy):
    """Skeleton: build -> compute_score -> clean -> judge, with ``target`` /
    ``direction`` / ``target_semantics`` instance state and a generic base
    ``_judge`` that maps score -> Verdict.

    ``target`` semantics (the polymorphic knob that replaces the 4-class
    split):
      - ``None``           -> measurement loop, always KEEP
      - ``'best'``         -> per-op-best comparison (KEEP if new best vs
                              ``state.per_op_best[op]`` or ``state.best_score``)
      - ``float``          -> absolute threshold (see ``target_semantics``)

    ``direction``:
      - ``'higher_better'`` -> score > state.best_score is improvement (cos,
                                top1, mAP)
      - ``'lower_better'``   -> score < state.best_score is improvement
                                (cos_loss = 1-cos, perplexity, mse)

    ``target_semantics`` (only meaningful when ``target`` is a float):
      - ``'ceiling'`` (default) -> ``target`` is the success goal; crossing it
        means ENOUGH (stop, success). Used by Phase 4 ``End2EndEval``
        (e.g. ``expected_acc=0.7``: stop when top1 >= 0.7).
      - ``'floor'``            -> ``target`` is the failure floor; falling
         below it means ENOUGH (stop, failure) or WITHDRAW. Used by
         cluster-acceptance (e.g.
         ``target_cos_threshold = baseline * expected_cos``: stop the search
         when a cluster's cos drops below the floor).
    """

    def __init__(self,
                 target: Optional[Any] = None,
                 direction: str = 'higher_better',
                 num_sample: int = 0,
                 expected: Optional[float] = None,
                 target_semantics: str = 'ceiling',
                 use_enough: bool = True):
        self.target = target
        self.direction = direction
        self.num_sample = num_sample
        self.target_semantics = target_semantics
        # use_enough: when False, the floor path returns WITHDRAW instead of
        # ENOUGH on failure (one-by-one retry tries all ops, doesn't stop).
        self.use_enough = use_enough
        self.expected = expected if expected is not None else (
            target if isinstance(target, (int, float)) else 0.0)

    def eval(self, trial: Trial, state: SearchState) -> Verdict:
        model = self._build_model(trial, state)
        try:
            score = self._compute_score(model, trial, state)
        finally:
            self._clean_model(model, trial, state)
        trial.score = score
        return self._judge(score, trial, state)

    def _build_model(self, trial: Trial, state: SearchState):
        if trial.mix_model is not None:
            return trial.mix_model
        assert trial.model_kwargs is not None, \
            "Trial must carry model_kwargs or mix_model for EvalBase"
        from .mix_precision import MixQuantModel
        return MixQuantModel(state.fp32_mlir, **trial.model_kwargs)

    def _clean_model(self, model, trial: Trial, state: SearchState) -> None:
        # Always clean: MixQuantModel.clean() is idempotent and safe whether
        # the model came from ``trial.mix_model`` or ``model_kwargs``.
        if hasattr(model, 'clean'):
            model.clean()

    def _record_min_loss(self, op: str, cos_loss: float, snr: float,
                         loss_dict: Dict[str, List[float]]) -> None:
        """Shared min(cos_loss, snr) tracking into ``loss_dict``.

        Records the first sample (extend) or tracks the min on subsequent
        samples. This is the single place the per-op sensitivity loss is
        accumulated for the cos-based search paths.
        """
        if op not in loss_dict:
            loss_dict[op].extend([cos_loss, snr])
        else:
            if cos_loss < loss_dict[op][0]:
                loss_dict[op][0] = cos_loss
            if snr < loss_dict[op][1]:
                loss_dict[op][1] = snr

    def _compute_score(self, model, trial: Trial, state: SearchState) -> float:
        raise NotImplementedError

    def _judge(self, score: float, trial: Trial, state: SearchState) -> Verdict:
        # Measurement loop: always KEEP.
        if self.target is None:
            return Verdict.KEEP

        # Per-op-best comparison. Reads ``state.per_op_best[op]`` which is
        # maintained by the TryStrategy on commit (PerOpThresholdTry's
        # _commit_unit writes it).
        if self.target == 'best':
            best = state.per_op_best.get(trial.op)
            if self.direction == 'higher_better':
                if best is None or score > best[0]:
                    return Verdict.KEEP
            else:
                if best is None or score < best[0]:
                    return Verdict.KEEP
            return Verdict.WITHDRAW

        # Absolute threshold. The two cases have opposite polarity on the
        # same (target, direction, score) signature:
        #   'ceiling' (default, End2EndEval): score >= target -> ENOUGH (stop, success)
        #   'floor'   (cluster acceptance):    score <  target -> ENOUGH (stop, failure)
        # 'floor': KEEP if score >= target, fail verdict if score < target.
        if self.target_semantics == 'floor':
            # use_enough=True (cluster acceptance): failure stops the loop.
            # use_enough=False (one-by-one retry): failure withdraws this op
            # and tries the next — never ENOUGH.
            fail_verdict = Verdict.ENOUGH if self.use_enough else Verdict.WITHDRAW
            if self.direction == 'higher_better':
                if score < self.target:
                    return fail_verdict
                return Verdict.KEEP
            else:  # lower_better (rare for floor — kept for symmetry)
                if score > self.target:
                    return fail_verdict
                return Verdict.KEEP
        # 'ceiling' (default) — success-goal: stop when target is reached.
        if self.direction == 'higher_better':
            if score >= self.target:
                return Verdict.ENOUGH
            if state.best_score is None or score > state.best_score:
                return Verdict.KEEP
            return Verdict.WITHDRAW
        else:  # lower_better
            if score <= self.target:
                return Verdict.ENOUGH
            if state.best_score is None or score < state.best_score:
                return Verdict.KEEP
            return Verdict.WITHDRAW


class CosEval(EvalBase):
    """All cos-based eval modes in one class, configured by ctor args:

      target='best',  direction='lower_better',  record_loss=False, cluster=False
          -> per-op best-threshold search
      target='best',  direction='lower_better',  record_loss=True,  cluster=True
          -> per-op best-threshold search with cluster loss recording
      target=None,    direction='lower_better',  record_loss=True
          -> per-op sensitivity sweep (low-prec / 4bit)
      target=None,    direction='lower_better',  record_loss=True,  sample_num=1
          -> per-op sensitivity sweep (fast search, single sample)
      target=<float>, direction='higher_better', record_loss=False
          -> cluster acceptance (use_enough flag controls ENOUGH vs WITHDRAW
          on threshold failure; default True)

    Score polarity:
      direction='lower_better'  -> _compute_score returns ``1 - cos`` (cos_loss)
      direction='higher_better' -> _compute_score returns raw ``cos``
    """

    def __init__(self,
                 target: Optional[Any],
                 direction: str,
                 mix_prec,
                 global_compare_layers,
                 layers_rate,
                 predictions_gt,
                 expected: float,
                 num_sample: int,
                 loss_methods: Optional[List[str]] = None,
                 sample_num: Optional[int] = None,
                 record_loss: bool = True,
                 cluster: bool = False,
                 use_enough: bool = True,
                 target_semantics: str = 'ceiling',
                 desc: str = 'cos'):
        super().__init__(target=target,
                         direction=direction,
                         num_sample=num_sample,
                         expected=expected,
                         target_semantics=target_semantics,
                         use_enough=use_enough)
        self.mix_prec = mix_prec
        self.logger = mix_prec.logger
        self.gcl = global_compare_layers
        self.lr = layers_rate
        self.gt = predictions_gt
        self.loss_methods = loss_methods or ['cos']
        self.sample_num = sample_num
        self.record_loss = record_loss
        self.cluster = cluster
        self.desc = desc
        self.loss_dict: Dict[str, List[float]] = collections.defaultdict(list)

    def _compute_score(self, model, trial: Trial, state: SearchState) -> float:
        # Absolute-threshold path: print cluster/layer info and stash snr on
        # the trial, return raw cos (higher better).
        if isinstance(self.target, (int, float)):
            result = self.mix_prec.run_model(model,
                                             False,
                                             self.gcl,
                                             self.lr,
                                             self.gt,
                                             loss_methods=['cos', 'snr'])
            trial.snr = result['snr']
            cos = result['cos']
            if trial.cluster is not None and len(trial.cluster) > 1:
                print(f'Cluster: {trial.cluster}, outputs_cos: {cos}, '
                      f'outputs_snr: {trial.snr} target_output_cos {self.target}')
            else:
                print(f'Layer: {trial.op}, outputs_cos: {cos}, '
                      f'outputs_snr: {trial.snr} target_output_cos {self.target}')
            return cos

        # Per-op-best / measurement path: returns cos_loss (lower better).
        # When record_loss, also tracks min(cos_loss, snr).
        result = self.mix_prec.run_model(model,
                                         False,
                                         self.gcl,
                                         self.lr,
                                         self.gt,
                                         sample_num=self.sample_num,
                                         loss_methods=self.loss_methods)
        if isinstance(result, dict):
            cos_loss = 1 - result['cos']
            snr = result.get('snr')
        else:
            # loss_methods=['cos'] returns a scalar
            cos_loss = 1 - result
            snr = None

        if self.record_loss and snr is not None:
            self._record_min_loss(trial.op, cos_loss, snr, self.loss_dict)
            # sensitivity-sweep wording
            self.logger.print_info("layer {}, outputs_cos:{}, outputs_snr:{}".format(
                trial.op, cos_loss, snr))
        else:
            # threshold-search wording
            self.logger.print_info("outputs_cos_los = {}".format(cos_loss))
        return cos_loss

    # _judge inherited from EvalBase — handles target=None/'best'/float with
    # target_semantics ('ceiling'/'floor') and use_enough (floor failure ->
    # ENOUGH when True, WITHDRAW when False for one-by-one retry).


class End2EndEval(EvalBase):
    """Phase 4 — task-level metric (top1, top5, mAP, perplexity, ...) on
    final model outputs via ``MixModelInference``.

    Same threshold-judge shape as ``CosEval(target=<float>,
    direction='higher_better')``; only the score source differs: instead of
    cos vs ref-tensors, ``_compute_score`` drives ``MixModelInference`` over
    a labeled dataset, returns the scalar metric (top1, mAP, ...) from
    ``score.get_result()``.

    Parse-once-reuse-per-trial: GT (label file, COCO json, ...) is parsed
    once in the ``MixModelInference.__init__`` (via the plugin's ``init()``);
    each trial calls ``set_model()`` + ``reset_score()`` (which clears
    per-trial counters via ``base_class.reset()`` while keeping the parsed
    GT). See ``docs/search-qtable-eval-merge-supplement.html`` §7.
    """

    def __init__(self,
                 target: Optional[Any],
                 direction: str,
                 runner_factory,
                 metric_index: int,
                 num_sample: int,
                 logger=None,
                 target_semantics: str = 'ceiling',
                 use_enough: bool = True):
        super().__init__(target=target,
                         direction=direction,
                         num_sample=num_sample,
                         expected=target if isinstance(target, (int, float)) else 0.0,
                         target_semantics=target_semantics,
                         use_enough=use_enough)
        # runner_factory: callable that returns a MixModelInference instance.
        # Deferred — called on first _compute_score, so MixModelInference
        # (which initializes cv2/oneDNN via preprocess()) doesn't interfere
        # with TPU model lowering during the search setup phase.
        self._runner_factory = runner_factory
        self._runner = None  # built lazily on first _compute_score
        self.metric_index = metric_index  # 0=top1, 1=top5 for topx
        self.logger = logger
        # loss_dict: compatibility with CosEval — the sensitivity sweep and
        # cluster_4_8 read eval_s.loss_dict[op] = [loss, snr] where both are
        # "lower = better". For end2end, we record [1 - score, 1 - score]
        # (higher score = lower loss = less sensitive op). When target is
        # 'best' or None (threshold/sensitivity modes), record_loss stays
        # False because _compute_score only records when target is a float
        # (cluster mode) — but the sensitivity call site accesses loss_dict
        # unconditionally, so we always populate it.
        self.loss_dict = collections.defaultdict(list)

    def _build_model(self, trial: Trial, state: SearchState):
        # Trial carries mix_model (cluster/low-prec/fast paths) or
        # model_kwargs (threshold path); build if needed.
        if trial.mix_model is not None:
            return trial.mix_model
        assert trial.model_kwargs is not None, \
            "Trial must carry model_kwargs or mix_model for End2EndEval"
        from .mix_precision import MixQuantModel
        return MixQuantModel(state.fp32_mlir, **trial.model_kwargs)

    def _clean_model(self, model, trial: Trial, state: SearchState) -> None:
        # Only clean models built here; mix_model is owned by the TryStrategy.
        if trial.mix_model is None and hasattr(model, 'clean'):
            model.clean()

    def _compute_score(self, model, trial: Trial, state: SearchState) -> float:
        # Lazily build the MixModelInference runner on first call.
        # This defers cv2/oneDNN initialization (via preprocess()) until
        # the first actual end2end eval, so it doesn't interfere with TPU
        # model lowering during the search setup phase.
        if self._runner is None:
            self._runner = self._runner_factory()
        # Forward the trial identity to the plugin's begin_model hook (tier-3
        # model-aware / self-sufficient plugins). n_fp/phase help a HIL plugin
        # know which model it is about to deploy.
        n_fp = None
        if trial.mix_table is not None and hasattr(trial.mix_table, '__len__'):
            n_fp = len(trial.mix_table)
        self._runner.set_model(model,
                               ctx_extra={
                                   'n_fp': n_fp,
                                   'phase': trial.op or trial.cluster or 'trial',
                               })
        # run_eval_trial owns reset -> begin_model -> [drive|self] ->
        # get_result -> end_model. A 'self' plugin runs its own loop.
        result = self._runner.run_eval_trial(self.num_sample)
        if isinstance(result, tuple):
            result = result[self.metric_index]
        # Record into loss_dict for the sensitivity sweep (cluster_4_8 reads
        # loss_dict[op] = [loss, snr]; for end2end we store [1 - score, 1 - score]
        # as a cos-loss-like pair where higher = more sensitive).
        if trial.op is not None:
            loss = 1.0 - result
            if trial.op not in self.loss_dict:
                self.loss_dict[trial.op].extend([loss, loss])
            else:
                if loss > self.loss_dict[trial.op][0]:
                    self.loss_dict[trial.op][0] = loss
                if loss > self.loss_dict[trial.op][1]:
                    self.loss_dict[trial.op][1] = loss
        if self.logger is not None:
            self.logger.print_info("[End2EndEval] trial={} score={:.4f} (metric_idx={})".format(
                trial.op or trial.cluster or 'n/a', result, self.metric_index))
        return result


class CustomEval:
    """Phase 6 — user-supplied fn returns ``Verdict`` directly, bypasses the
    ``EvalBase`` skeleton.

    TODO (Phase 6): implement with ``user_fn(trial, outputs, state) -> Verdict``.
    """

    def __init__(self, user_fn, target: Optional[Any] = None, direction: str = 'higher_better'):
        self.user_fn = user_fn
        raise NotImplementedError("CustomEval is Phase 6 — not yet implemented")


# ==============================================================================
# Score-plugin resolution: built-in name OR external customer file.
# ==============================================================================


def resolve_score_plugin(spec: str, config_dir: str = ''):
    """Resolve the score-plugin class from a ``spec`` string.

    Two accepted forms (docs/custom-end2end-eval-plugin-proposal.md §3):

    - bare built-in name, e.g. ``topx`` / ``coco_mAP``: package-relative
      import from ``eval.postprocess_and_score_calc``.
    - external file, e.g. ``./my_metric.py::MyMetric``: loaded via
      ``importlib`` from the file path. ``ClassName`` after ``::`` defaults
      to the file stem when omitted. Relative paths resolve against
      ``config_dir`` (the eval-config JSON directory) first, then CWD.

    After loading an external class, the 6-method plugin contract
    (init / preproc / update / get_result / print_info / reset) is verified
    so a broken plugin fails here with a clear message instead of mid-search.
    """
    import importlib.util
    import os

    external = ('::' in spec or spec.endswith('.py') or os.sep in spec
                or (os.altsep is not None and os.altsep in spec))
    if not external:
        namespace = {}
        exec(f'from eval.postprocess_and_score_calc.{spec} import {spec}', namespace)
        return namespace[spec]

    path, _, cls_name = spec.partition('::')
    if os.path.isabs(path):
        candidates = [path]
    else:
        candidates = []
        if config_dir:
            candidates.append(os.path.join(config_dir, path))
        candidates.append(path)  # CWD-relative fallback
    resolved = next((c for c in candidates if os.path.isfile(c)), None)
    if resolved is None:
        raise RuntimeError("score_plugin file '{}' not found; tried: {}".format(spec, candidates))
    mod_name = os.path.splitext(os.path.basename(resolved))[0]
    mod_spec = importlib.util.spec_from_file_location(mod_name, resolved)
    module = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(module)
    cls_name = cls_name or mod_name
    cls = getattr(module, cls_name, None)
    if cls is None:
        raise RuntimeError("score_plugin class '{}' not found in '{}' (class name must match the "
                           "part after '::' or the file stem)".format(cls_name, resolved))
    for method in ('init', 'preproc', 'update', 'get_result', 'print_info', 'reset'):
        if not hasattr(cls, method):
            raise RuntimeError("score_plugin class '{}' lacks required method '{}()'".format(
                cls_name, method))
    # Capability validation: data_mode must be known; a 'plugin' (tier-2)
    # data plugin must also be iterable (its own sample enumeration).
    data_mode = getattr(cls, 'data_mode', 'toolkit')
    if data_mode not in ('toolkit', 'plugin'):
        raise RuntimeError(
            "score_plugin class '{}' has invalid data_mode '{}' (expected 'toolkit' or "
            "'plugin')".format(cls_name, data_mode))
    if data_mode == 'plugin' and not hasattr(cls, '__iter__'):
        raise RuntimeError("score_plugin class '{}' sets data_mode='plugin' but does not implement "
                           "__iter__ (required to enumerate its own samples)".format(cls_name))
    print('[score_plugin] loaded external plugin {} from {}'.format(cls_name, resolved))
    return cls


# ==============================================================================
# MixModelInference — bridge between the search-loop in-memory MixQuantModel
# and the model_eval.py scoring pipeline.
# ==============================================================================
#
# Lazy loading: common_inference lives in python/eval/model_inference.py which
# imports pymlir + cv2 at module level. We don't want those pulled in just to
# import the cos eval classes (EvalBase / CosEval / End2EndEval). MixModelInference
# is only needed on the end-to-end path (when --eval_config is used); the cos
# path (default) never touches it.
#
# We use a module-level __getattr__ so that:
#   from calibration.cali_eval import CosEval        # works without pymlir
#   from calibration.cali_eval import MixModelInference  # triggers lazy build
#   MixModelInference(args, model)                  # calls the real class


def _build_mix_model_inference_cls():
    """Build the MixModelInference class lazily, after common_inference is
    importable. Called on first ``from calibration.cali_eval import
    MixModelInference`` via module __getattr__."""
    from eval.model_inference import common_inference
    from utils.preprocess import preprocess
    from utils.misc import parse_debug_cmd

    class MixModelInference(common_inference):
        """Bridges an in-memory ``MixQuantModel`` (from the search loop) into the
        ``model_eval.py`` scoring pipeline.

        Reuses ``img_proc`` / ``score`` / ``model_invoke`` / ``get_result`` from
        :class:`common_inference`; overrides what's needed to swap the model per
        trial:

        - ``__init__`` skips the file load (the ``MixQuantModel`` already has its
          ``pymlir.module`` loaded + its ``MlirParser``); sets up ``img_proc``
          and the score plugin (``topx`` / ``coco_mAP`` / custom) once.
        - ``invoke`` uses ``mix_model.infer(self.x, None)`` instead of
          ``self.module.set_tensor`` + ``invoke`` + ``get_all_tensor``.
        - ``set_model`` swaps the in-memory model between trials.
        - ``reset_score`` clears per-trial counters in the score plugin via the
          plugin's ``reset()`` override (parsed GT from ``init()`` is kept).
        - ``get_sample`` builds the (img_path, label) list once via
          :class:`DataSelector` so each trial reuses the same sample sequence.

        Lifetime: constructed once per search (img_proc + score plugin set up
        once); ``set_model(mix_model)`` swaps the in-memory module per trial;
        ``reset_score()`` clears per-trial counters while keeping GT parsed in
        ``init()`` (label file, COCO json, etc.). See the parse-once-reuse-per-
        trial pattern in ``docs/search-qtable-eval-merge-supplement.html`` §7.

        See ``python/calibration/end2end_topx_example.py`` for a runnable
        reference that doubles as the customer template.
        """

        def __init__(self, args, mix_model):
            # Bypass common_inference.__init__'s file load; reuse mix_model.
            self.args = args
            self.idx = 0
            self.postprocess_type = args.postprocess_type
            self.use_cuda = mix_model.using_cuda
            self._mix_model = mix_model
            self.module = mix_model.module  # shared loaded pymlir.module
            self.module_parsered = mix_model.parser  # shared MlirParser
            self.batch_size = self.module_parsered.get_batch_size()
            args.batch_size = self.batch_size
            self.input_num = self.module_parsered.get_input_num()
            self.img_proc = preprocess(args.debug_cmd)
            self.img_proc.load_config(self.module_parsered.get_input_op_by_idx(0))
            args.net_input_dims = self.img_proc.net_input_dims
            self.batched_labels = []
            self.batched_imgs = ''
            # Import + instantiate the score plugin. Bare names ('topx',
            # 'coco_mAP') resolve to the built-in package; a 'path::Class'
            # spec loads an external customer file (wheel workflow — see
            # docs/custom-end2end-eval-plugin-proposal.md).
            config_dir = getattr(args, 'eval_config_dir', '') or ''
            self.score_cls = resolve_score_plugin(args.postprocess_type, config_dir)
            self.score = self.score_cls(args)
            # debug_cmd is already a dict (parsed by run_calibration.py);
            # keep as-is rather than re-parsing (parse_debug_cmd expects a str).
            self.debug_cmd = args.debug_cmd if isinstance(args.debug_cmd, dict) \
                             else parse_debug_cmd(args.debug_cmd)
            print(f'[MixModelInference] batch_size={self.batch_size} '
                  f'plugin={args.postprocess_type}')

        def set_model(self, mix_model, ctx_extra=None):
            """Swap the in-memory model for the next trial. ``img_proc`` and
            ``score`` config are NOT rebuilt (preprocess doesn't change across
            trials).

            ``ctx_extra`` (dict, optional): static context the caller wants
            forwarded to the plugin's ``begin_model`` (e.g. ``cali_table``,
            ``n_fp``, ``phase``) that the bridge cannot infer from the model
            object alone. Merged into ``model_ctx`` in ``run_eval_trial``.
            """
            self._mix_model = mix_model
            self.module = mix_model.module
            self.module_parsered = mix_model.parser
            self.use_cuda = mix_model.using_cuda  # keep consistent with swapped model
            self._ctx_extra = ctx_extra or {}

        def _build_model_ctx(self):
            """Assemble the per-trial ``model_ctx`` handed to
            ``plugin.begin_model``. Everything the bridge can see about the
            model under test, plus whatever the caller passed via
            ``set_model(ctx_extra=...)``."""
            import types as _types
            m = self._mix_model
            ctx = dict(
                chip=getattr(m, 'chip', None),
                fp32_mlir=getattr(m, 'fp32_mlir', None),
                quanted_mlir_file=getattr(m, 'quanted_mlir_file', None),
                weight_file=getattr(m, 'weight_file', None),
                cali_table=getattr(m, 'calib_table', None),
                mix_table=getattr(m, 'mix_table', None),
                mode=getattr(m, 'mode', None),
                high_prec=getattr(m, 'high_prec', None),
                using_cuda=getattr(m, 'using_cuda', False),
                module=getattr(m, 'module', None),
                parser=getattr(m, 'parser', None),
            )
            ctx.update(getattr(self, '_ctx_extra', {}) or {})
            return _types.SimpleNamespace(**ctx)

        def run_eval_trial(self, n, progress_cb=None):
            """Own one trial's eval lifecycle (the single driver both call
            sites share):

                reset() -> begin_model(model_ctx) -> [drive samples | self]
                        -> get_result() -> end_model()

            Three modes:
            - ``begin_model`` returns ``'self'`` (tier-3): the plugin already
              ran its own full loop (its dataset + preprocess + inference);
              the harness skips driving. ``begin_model`` runs AFTER
              ``reset()`` so its accumulation survives.
            - plugin ``data_mode == 'plugin'`` (tier-2): the harness enumerates
              the plugin's OWN samples (``__iter__``) and preprocess
              (``preproc(sample)``) but still runs simulator inference.
            - default (tier-1): the harness owns dataset (ImageFolder /
              DataSelector), preprocess (``img_proc``) and enumeration.

            ``progress_cb(idx, count)`` is invoked after each sample.
            Returns the plugin's ``get_result()`` (the caller applies
            ``metric_index``).
            """
            self.reset_score()
            mode = None
            begin = getattr(self.score, 'begin_model', None)
            if callable(begin):
                mode = begin(self._build_model_ctx())
            if mode == 'self':
                pass  # plugin ran its own loop inside begin_model
            elif getattr(self.score, 'data_mode', 'toolkit') == 'plugin':
                self._run_plugin_data_loop(n, progress_cb)
            else:
                self._run_toolkit_loop(n, progress_cb)
            result = self.get_result()
            end = getattr(self.score, 'end_model', None)
            if callable(end):
                end()
            return result

        def _run_toolkit_loop(self, n, progress_cb):
            """tier-1: harness-owned dataset + img_proc + enumeration."""
            for idx in range(n):
                img_path, label = self.get_sample(idx)
                if img_path is None:
                    break
                self._run_one(idx, img_path, label, lambda: self.run(idx, img_path, label))
                if progress_cb is not None:
                    progress_cb(idx, getattr(self.score, 'idx', idx + 1))

        def _run_plugin_data_loop(self, n, progress_cb):
            """tier-2: customer-owned dataset (__iter__) + preprocess
            (preproc), harness-owned simulator inference. ``open_dataset`` is
            called once (first trial); ``__iter__`` is re-run each trial and
            must yield the same sequence. The customer's ``update`` receives
            the raw ``sample`` as its 3rd positional arg."""
            if not getattr(self, '_dataset_opened', False):
                od = getattr(self.score, 'open_dataset', None)
                if callable(od):
                    od(self.args)
                self._dataset_opened = True
            count = 0
            for sample in self.score:  # plugin.__iter__ — fresh each trial
                if n and count >= n:
                    break
                self._run_one(count, sample, None, lambda: self._builtin_infer(count, sample))
                if progress_cb is not None:
                    progress_cb(count, getattr(self.score, 'idx', count + 1))
                count += 1

        def default_preprocess(self, sample):
            """The harness's common preprocess (the mlir's img_proc), exposed
            so ``eval_sample`` can reuse it: ``preprocess(sample)`` -> model
            input tensor. ``sample`` is an image path."""
            return self.img_proc.run(sample)

        def _run_one(self, idx, sample, label, builtin_fn):
            """Per-sample dispatch shared by both loops. If the plugin overrides
            ``eval_sample`` (returns non-None predictions), the customer owns
            preprocess+inference+postprocess and we accumulate host-side via
            update(predictions). Otherwise fall back to the built-in path
            (img_proc + simulator -> raw outputs -> update(outputs))."""
            ev = getattr(self.score, 'eval_sample', None)
            predictions = ev(sample, self.default_preprocess) if callable(ev) else None
            if predictions is not None:
                # Host-side: predictions vs GT (in the plugin's update).
                self.score.update(idx, predictions, sample)
            else:
                builtin_fn()

        def _builtin_infer(self, idx, sample):
            """tier-2 built-in inference: plugin.preproc(sample) -> simulator
            invoke -> update(outputs). Used when eval_sample is not overridden
            but the plugin still supplies its own preprocess."""
            self.x = self.score.preproc(sample)
            outputs = self.invoke()
            self.score.update(idx, outputs, sample)

        def reset_score(self):
            """Clear per-trial counters in the score plugin (topx.c1/c5/idx,
            coco_mAP.json_dict/ratio_list). GT parsed in ``init()`` is kept
            via the plugin's ``reset()`` override."""
            self.score.reset()

        def get_sample(self, idx):
            """Return (img_path, label) for sample idx. The sample list is
            built once via :class:`DataSelector` on the dataset path; labels
            come from the score plugin's parsed GT (topx reads --label_file)
            or a parallel label list.

            Returns ``(img_path, None)`` when no per-sample label is available
            (the plugin falls back to filename matching in ``update``).
            """
            if not hasattr(self, '_samples'):
                self._build_sample_list()
            if idx >= len(self._samples):
                return None, None
            item = self._samples[idx]
            if isinstance(item, tuple):
                return item
            return item, None

        def _build_sample_list(self):
            """Build the sample list [(img_path, label_or_None), ...] once.

            PRIMARY mechanism — ALIGNED TO ``model_eval.py`` EXACTLY: when the
            dataset is a folder-per-class tree (``<dataset>/<classdir>/<img>``),
            select images the SAME way ``model_eval.py --dataset_type imagenet``
            does — via torchvision ``ImageFolder`` with ``shuffle=False`` — i.e.
            the sequential ``ImageFolder.imgs`` order, first ``count`` images,
            with the label = alphabetical class index. This makes the search's
            end2end eval score the IDENTICAL images (and labels) as the deploy
            ``model_eval.py``, so the two agree. No DataSelector random shuffle,
            no val.txt.

            FALLBACK (only when the dataset is NOT a folder-per-class tree, i.e.
            ImageFolder finds no class subdirs): use DataSelector + the score
            plugin's parsed label file (all_labels_dict / all_labels_list), else
            pass None.

            COCO DETECTION (coco_mAP): a flat image dir; select images the SAME
            way ``model_eval.py --dataset_type coco`` does — via
            ``get_image_list(dataset, count)`` (os.listdir first-N) — and pass
            label=None (detection GT comes from the COCO annotation json, not
            per-image labels; coco_mAP.update derives image_id from the path).
            """
            import os
            import pathlib

            count = self.args.count or 0

            # --- COCO DETECTION: flat dir, GT via annotation json (== model_eval) --
            if str(getattr(self, 'postprocess_type', '')) == 'coco_mAP':
                from utils.misc import get_image_list
                img_list = get_image_list(self.args.dataset, count)
                self._samples = [(p, None) for p in img_list]
                self._log("[MixModelInference] subset: get_image_list first-N "
                          "(== model_eval.py --dataset_type coco), {} images; labels: COCO "
                          "annotation json (per-image label=None)".format(len(img_list)))
                return

            # --- PRIMARY: torchvision ImageFolder, sequential (== model_eval) --
            dataset_root = self.args.dataset or ''
            folder = None
            if dataset_root and os.path.isdir(dataset_root):
                try:
                    import torchvision.datasets as datasets
                    folder = datasets.ImageFolder(dataset_root)
                except Exception as e:
                    self._log("[MixModelInference] ImageFolder unavailable ({}), "
                              "falling back to DataSelector".format(e))
                    folder = None

            if folder is not None and len(getattr(folder, 'imgs', [])) > 0:
                # folder.imgs is the ordered [(path, class_idx), ...] list that
                # torchvision.DataLoader(shuffle=False) iterates; model_eval
                # takes the first `count`. Replicate exactly.
                imgs = folder.imgs
                if count and count < len(imgs):
                    imgs = imgs[:count]
                samples = [(p, int(lbl)) for (p, lbl) in imgs]
                # topx.update prefers an explicit passed label; clear any parsed
                # label-file state so our ImageFolder label is used.
                if hasattr(self.score, 'all_labels_dict'):
                    self.score.all_labels_dict = {}
                if hasattr(self.score, 'all_labels_list'):
                    self.score.all_labels_list = []
                self._samples = samples
                self._log("[MixModelInference] subset+labels: torchvision ImageFolder "
                          "sequential (shuffle=False, == model_eval.py), {} classes, {} images "
                          "(DataSelector/val.txt not used)".format(len(folder.classes),
                                                                   len(samples)))
                return

            # --- FALLBACK: not a folder-per-class tree ----------------------
            from calibration.data_selector import DataSelector
            ds = DataSelector(self.args.dataset, self.args.count, self.args.data_list)
            img_paths = ds.data_list
            labels_dict = getattr(self.score, 'all_labels_dict', {}) or {}
            labels_list = getattr(self.score, 'all_labels_list', []) or []

            if labels_dict:
                # Order-independent filename match — topx.update handles it.
                self._samples = [(p, None) for p in img_paths]
                self._log("[MixModelInference] subset: DataSelector; labels: label_file dict "
                          "(no ImageFolder class subdirs found)")
                return

            if labels_list:
                exts = ('.jpg', '.jpeg', '.png', '.bmp')
                full = sorted(
                    str(f) for f in pathlib.Path(self.args.dataset).glob('**/*')
                    if str(f).lower().endswith(exts))
                fname_to_label = {
                    fp: labels_list[i]
                    for i, fp in enumerate(full) if i < len(labels_list)
                }
                self._samples = [(p, fname_to_label.get(p)) for p in img_paths]
                self.score.all_labels_list = []
                self._log("[MixModelInference] subset: DataSelector; labels: label_file list")
                return

            # No label info at all — pass None (topx errors if it needs one).
            self._samples = [(p, None) for p in img_paths]
            self._log("[MixModelInference] subset: DataSelector; labels: none resolved "
                      "(no ImageFolder, no label_file) — passing None")

        def _log(self, msg):
            lg = getattr(self, 'logger', None)
            if lg is not None and hasattr(lg, 'print_info'):
                lg.print_info(msg)
            else:
                print(msg)

        def invoke(self):
            # MixQuantModel.infer(data, None) expects data as a LIST (one
            # array per model input, zipped with module.input_names).
            # self.x from img_proc.run() is a single preprocessed array —
            # wrap it in a list so infer() sees one input, not an iteration
            # over the batch axis (which would set a wrong-shaped tensor and
            # make the TPU fail to create the convolution primitive).
            x = self.x
            # CUDA path: MixQuantModel.infer -> set_tensor(k, v) (no shape)
            # expects float32 (matches mlir_inference.invoke's CUDA branch).
            if getattr(self._mix_model, 'using_cuda', False) and hasattr(x, 'astype'):
                x = x.astype('float32')
            data = x if isinstance(x, (list, tuple)) else [x]
            outputs_dict = self._mix_model.infer(data, None)
            return list(outputs_dict.values())

    return MixModelInference


def __getattr__(name):
    """Module-level lazy loader for ``MixModelInference``.

    ``from calibration.cali_eval import CosEval`` — works without pymlir/cv2
    (CosEval is defined at module level).

    ``from calibration.cali_eval import MixModelInference`` — triggers this
    hook, builds the real class (which imports common_inference → pymlir),
    caches it in globals, and returns it. Subsequent accesses hit the cache.
    """
    if name == 'MixModelInference':
        cls = _build_mix_model_inference_cls()
        globals()['MixModelInference'] = cls  # cache: next access skips __getattr__
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
