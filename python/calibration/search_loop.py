#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
"""Unified adjust->eval->retry loop for search-qtable.

Implements the SearchLoop skeleton + TryStrategy abstraction (the "ask" /
adjust half of ask/tell). The EvalStrategy / EvalBase / CosEval axis lives
in ``python/calibration/cali_eval.py`` — see ``docs/search-qtable-eval-
merge-supplement.html`` and ``docs/search-qtable-unified-eval-proposal.html``
for the full design.

Proven patterns: ask/tell (Optuna, scikit-optimize, SMAC3), Iterated Local
Search (Lourenco 2003), Strategy + Template Method (Gamma 1994), Memento /
Command for commit/rollback, tri-state acceptance criterion.
"""
import gc
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from tqdm import tqdm


class Verdict(Enum):
    """Tri-state acceptance protocol returned by ``EvalStrategy.eval``.

    KEEP     - accept the trial, commit it, continue searching.
    WITHDRAW - reject the trial, rollback, continue searching.
    ENOUGH   - stop the loop (target met / no progress / budget spent).
    """
    KEEP = 1
    WITHDRAW = 2
    ENOUGH = 3


@dataclass
class Trial:
    """A single trial proposed by a ``TryStrategy``.

    Carries everything the EvalStrategy needs to build/score the model and
    everything the TryStrategy needs to commit/rollback. Concrete
    TryStrategies populate the fields they need; unused fields stay None.
    """
    op: Optional[str] = None
    method: Optional[str] = None
    cluster: Optional[List[str]] = None
    round: Optional[int] = None
    threshold: Optional[float] = None
    op_idx: Optional[int] = None
    method_idx: Optional[int] = None
    mix_table: Optional[str] = None
    mix_model: Optional[Any] = None
    model_kwargs: Optional[Dict[str, Any]] = None
    score: Optional[float] = None
    verdict: Optional[Verdict] = None


@dataclass
class SearchState:
    """Mutable search state shared between ``SearchLoop`` and the strategies.

    ``searcher`` lets strategies reach state still held by the owning
    ``SearchQtable*`` instance (cali-table helpers, args, etc.).
    """
    fp32_mlir: str
    chip: Optional[str] = None
    cali_table_name: Optional[str] = None
    qtable: Optional[Any] = None
    mix_table: Optional[str] = None
    predictions_gt: List[Any] = field(default_factory=list)
    best_score: Optional[float] = None
    expected_cos: float = 0.99
    target_cos_threshold: Optional[float] = None
    baseline_cos: Optional[float] = None
    num_sample: int = 0
    layer_types: Dict[str, str] = field(default_factory=dict)
    per_op_best: Dict[str, List[float]] = field(default_factory=dict)
    loss_dict: Dict[str, List[float]] = field(default_factory=dict)
    searcher: Optional[Any] = None
    ctx: Dict[str, Any] = field(default_factory=dict)


class TryStrategy(ABC):
    """How the qtable/cali-table is adjusted - the 'ask' half of ask/tell."""

    @abstractmethod
    def propose(self, state: SearchState) -> Optional[Trial]:
        """Return the next trial to evaluate, or None to skip this unit
        (the loop will advance past it). Return None when exhausted to stop.
        """

    @abstractmethod
    def commit(self, trial: Trial, state: SearchState) -> None:
        """Accept the trial into state (called on KEEP)."""

    @abstractmethod
    def rollback(self, trial: Trial, state: SearchState) -> None:
        """Undo the trial's state mutation (called on WITHDRAW / ENOUGH)."""

    def advance(self, trial: Trial, state: SearchState) -> None:
        """Move the internal cursor past this trial. Called after every trial
        regardless of verdict. Default: no-op (cursor managed in propose)."""

    @abstractmethod
    def exhausted(self, state: SearchState) -> bool:
        """True when no more trials can be proposed."""

    def num_units(self, state: SearchState) -> int:
        """Total unit count for the progress bar. Default: 0 (no pbar)."""
        return 0


class CursorTryStrategy(TryStrategy):
    """Base for strategies that walk an ordered sequence of trial units.

    A 'unit' is whatever the derived class defines: an (op, method) pair for
    threshold search, a cluster of ops for cluster-acceptance, a single op
    for float-search. The base handles cursor movement and skip-on-None;
    the derived class handles unit-level propose/commit/rollback.

    State-mutation style is agnostic: ``_propose_unit`` may mutate state
    optimistically (Memento pattern - cluster: propose removes ops, rollback
    restores) or be read-only (Command pattern - threshold: propose builds a
    mix_table, commit updates per-op-best). Both patterns are valid.
    """

    def __init__(self, state: SearchState):
        self._cursor = 0

    @abstractmethod
    def _num_units(self, state: SearchState) -> int:
        ...

    @abstractmethod
    def _propose_unit(self, idx: int, state: SearchState) -> Optional[Trial]:
        """Return the trial for unit at idx, or None to skip it."""

    @abstractmethod
    def _commit_unit(self, trial: Trial, idx: int, state: SearchState) -> None:
        ...

    @abstractmethod
    def _rollback_unit(self, trial: Trial, idx: int, state: SearchState) -> None:
        ...

    def propose(self, state: SearchState) -> Optional[Trial]:
        while self._cursor < self._num_units(state):
            trial = self._propose_unit(self._cursor, state)
            if trial is not None:
                return trial
            self._cursor += 1
        return None

    def commit(self, trial: Trial, state: SearchState) -> None:
        self._commit_unit(trial, self._cursor, state)

    def rollback(self, trial: Trial, state: SearchState) -> None:
        self._rollback_unit(trial, self._cursor, state)

    def advance(self, trial: Trial, state: SearchState) -> None:
        self._cursor += 1

    def exhausted(self, state: SearchState) -> bool:
        return self._cursor >= self._num_units(state)

    def num_units(self, state: SearchState) -> int:
        return self._num_units(state)


class EvalStrategy(ABC):
    """How a trial is scored and judged - the 'tell' half of ask/tell."""

    @abstractmethod
    def eval(self, trial: Trial, state: SearchState) -> Verdict:
        """Score the trial and return KEEP / WITHDRAW / ENOUGH."""


class SearchLoop:
    """The skeleton: propose -> eval -> commit/rollback/advance, with a
    tri-state Verdict protocol and an outer max_trials budget guard.

    Recursable: a ``TryStrategy.advance()`` may run a nested ``SearchLoop``
    (e.g. cluster one-by-one retry, fast-search round sweep). The nested loop
    shares ``SearchState`` but gets its own ``TryStrategy`` cursor.
    """

    def __init__(self,
                 try_s: TryStrategy,
                 eval_s: EvalStrategy,
                 desc: str = 'search',
                 max_trials: Optional[int] = None,
                 show_pbar: bool = True,
                 pbar_position: Optional[int] = None,
                 pbar_leave: bool = True,
                 pbar_bar_format: Optional[str] = None):
        self.try_s = try_s
        self.eval_s = eval_s
        self.desc = desc
        self.max_trials = max_trials
        self.show_pbar = show_pbar
        self.pbar_position = pbar_position
        self.pbar_leave = pbar_leave
        self.pbar_bar_format = pbar_bar_format

    def run(self, state: SearchState) -> SearchState:
        total = self.try_s.num_units(state) if self.show_pbar else None
        pbar = tqdm(total=total,
                    desc=self.desc,
                    position=self.pbar_position,
                    leave=self.pbar_leave,
                    bar_format=self.pbar_bar_format) if total else None
        n = 0
        try:
            while not self.try_s.exhausted(state):
                if self.max_trials is not None and n >= self.max_trials:
                    break
                trial = self.try_s.propose(state)
                if trial is None:
                    break
                try:
                    verdict = self.eval_s.eval(trial, state)
                except Exception:
                    self.try_s.rollback(trial, state)
                    raise
                trial.verdict = verdict
                if verdict == Verdict.KEEP:
                    self.try_s.commit(trial, state)
                    if trial.score is not None and \
                       (state.best_score is None or trial.score > state.best_score):
                        state.best_score = trial.score
                elif verdict == Verdict.WITHDRAW:
                    self.try_s.rollback(trial, state)
                elif verdict == Verdict.ENOUGH:
                    self.try_s.rollback(trial, state)
                    self.try_s.advance(trial, state)
                    break
                self.try_s.advance(trial, state)
                n += 1
                if pbar is not None:
                    if trial.op is not None and trial.score is not None:
                        cos_val = 1 - trial.score if trial.score is not None else 0.0
                        method_str = " | m={}".format(trial.method) \
                            if trial.method is not None else ""
                        pbar.set_postfix_str("{}{} cos={:.4f}".format(trial.op, method_str,
                                                                      cos_val))
                    pbar.update(1)
        finally:
            if pbar is not None:
                pbar.close()
        return state


class PerOpThresholdTry(CursorTryStrategy):
    """Sensitive-layer / threshold search (Phase 2).

    Walks the (ops x methods) grid as a flat cursor; tracks per-op best
    (lowest cos-loss = highest cos); finalizes each op on op-boundary by
    writing the best threshold via ``set_th_fn`` and recording into
    ``sensitive_layer_analysis_dict`` (unless the op is float).
    """

    def __init__(self, state: SearchState, ops: List[str], methods: List[str], layer_th_dicts: Dict,
                 set_th_fn, isolate: bool, mix_prec, qtable, high_prec_type, chip, low_prec,
                 high_prec, classifier, parser):
        super().__init__(state)
        self.ops = ops
        self.methods = methods
        self.layer_th_dicts = layer_th_dicts
        self.set_th_fn = set_th_fn
        self.isolate = isolate
        self.mix_prec = mix_prec
        self.logger = mix_prec.logger
        self.qtable = qtable
        self.high_prec_type = high_prec_type
        self.chip = chip
        self.low_prec = low_prec
        self.high_prec = high_prec
        self.classifier = classifier
        self.parser = parser
        self.fp_layer_list = list(ops)
        self.sensitive_layer_analysis_dict: Dict[str, List] = {}

    def _num_units(self, state: SearchState) -> int:
        return len(self.ops) * len(self.methods)

    def _flat_to_grid(self, idx: int):
        return idx // len(self.methods), idx % len(self.methods)

    def _propose_unit(self, idx: int, state: SearchState) -> Optional[Trial]:
        op_idx, method_idx = self._flat_to_grid(idx)
        op = self.ops[op_idx]
        if self.classifier.skip_threshold_search(op):
            return None
        method = self.methods[method_idx]
        threshold = self.layer_th_dicts[method][op][1]
        layer_type = self.parser.get_op_type_by_op_name(op)
        if method_idx == 0:
            self.logger.print_info("start to handle layer: {}, type: {}".format(op, layer_type))
        self.logger.print_info("adjust layer {} th, with method {}, and threshlod {}".format(
            op, method, threshold))
        if op in self.fp_layer_list:
            self.fp_layer_list.remove(op)
        if self.isolate:
            mix_table = self.mix_prec._gen_mix_table(self.fp_layer_list,
                                                     self.qtable,
                                                     high_prec_type=self.high_prec_type)
        else:
            mix_table = self.mix_prec._gen_mix_table([],
                                                     self.qtable,
                                                     high_prec_type=self.high_prec_type)
        new_cali_table_name = self.set_th_fn(op, threshold)
        state.cali_table_name = new_cali_table_name
        model_kwargs = dict(chip=self.chip,
                            mode=self.low_prec,
                            high_prec=self.high_prec,
                            calib_table=new_cali_table_name,
                            mix_table=mix_table,
                            using_cuda=True)
        return Trial(op=op,
                     method=method,
                     threshold=threshold,
                     op_idx=op_idx,
                     method_idx=method_idx,
                     mix_table=mix_table,
                     model_kwargs=model_kwargs)

    def _commit_unit(self, trial: Trial, idx: int, state: SearchState) -> None:
        op = trial.op
        state.per_op_best[op] = [trial.score, trial.threshold, trial.method]

    def _rollback_unit(self, trial: Trial, idx: int, state: SearchState) -> None:
        pass

    def advance(self, trial: Trial, state: SearchState) -> None:
        self._cursor += 1
        if self._cursor % len(self.methods) == 0 and \
           self._cursor <= self._num_units(state):
            op_idx = (self._cursor - 1) // len(self.methods)
            self._finalize_op(op_idx, state)

    def _finalize_op(self, op_idx: int, state: SearchState) -> None:
        op = self.ops[op_idx]
        if op not in self.fp_layer_list:
            self.fp_layer_list.append(op)
        if op in state.per_op_best:
            best = state.per_op_best[op]
            state.cali_table_name = self.set_th_fn(op, best[1])
            if not self.classifier.is_float(op):
                layer_type = self.parser.get_op_type_by_op_name(op)
                self.sensitive_layer_analysis_dict[op] = [best[0], layer_type]
                self.logger.print_info(
                    "layer {}, layer type is {}, best_th = {}, best_method = {}, best_cos_loss = {}"
                    .format(op, layer_type, best[1], best[2], best[0]))
        gc.collect()


class LowPrecSensitivityTry(CursorTryStrategy):
    """Per-op sensitivity sweep for 4bit search (Part 2).

    Demotes each op one at a time (removes from keep_int8_layers so it
    goes to low_prec while the rest stay at high_prec), evals cos+snr,
    records into loss_dict, then restores the op. Always returns KEEP;
    the restore happens in advance() (idempotent via rollback()).
    """

    def __init__(self, state: SearchState, ops: List[str], keep_int8_layers: List[str], strategy,
                 mix_prec, qtable, classifier, parser):
        super().__init__(state)
        self.ops = ops
        self.keep_int8_layers = keep_int8_layers
        self.strategy = strategy
        self.mix_prec = mix_prec
        self.logger = mix_prec.logger
        self.qtable = qtable
        self.classifier = classifier
        self.parser = parser

    def _num_units(self, state: SearchState) -> int:
        return len(self.ops)

    def _propose_unit(self, idx: int, state: SearchState) -> Optional[Trial]:
        op = self.ops[idx]
        if self.classifier.skip_sensitive_search(op):
            return None
        layer_type = self.parser.get_op_type_by_op_name(op)
        self.logger.print_info("start to handle layer: {}, type: {}".format(op, layer_type))
        self.keep_int8_layers.remove(op)
        self.strategy.before_trial(state.searcher, op)
        high_prec_type = self.strategy.qtable_keep_mode(state.searcher)
        mix_table = self.mix_prec._gen_mix_table(mix_ops=self.keep_int8_layers,
                                                 qtable=self.qtable,
                                                 high_prec_type=high_prec_type)
        mix_model = self.strategy.make_trial_model(state.searcher, mix_table)
        return Trial(op=op, mix_model=mix_model, mix_table=mix_table)

    def _commit_unit(self, trial: Trial, idx: int, state: SearchState) -> None:
        pass

    def _rollback_unit(self, trial: Trial, idx: int, state: SearchState) -> None:
        self._restore_op(trial, state)

    def advance(self, trial: Trial, state: SearchState) -> None:
        self._restore_op(trial, state)
        self._cursor += 1

    def _restore_op(self, trial: Trial, state: SearchState) -> None:
        if trial.op not in self.keep_int8_layers:
            self.keep_int8_layers.append(trial.op)
        self.strategy.rollback_trial(state.searcher, trial.op)


class FastSensitivityTry(CursorTryStrategy):
    """Per-op sensitivity sweep for fast search (Phase 5a).

    For each candidate op, temporarily quantize it (remove from fp_layer_list
    so it drops to low_prec while the rest stay float), measure cos+snr loss,
    record into loss_dict, then restore. Always KEEP (measurement loop).
    """

    def __init__(self, state: SearchState, ops: List[str], fp_layer_list: List[str], mix_prec,
                 qtable, chip, low_prec, high_prec, cali_table_name, parser):
        super().__init__(state)
        self.ops = ops
        self.fp_layer_list = fp_layer_list
        self.mix_prec = mix_prec
        self.logger = mix_prec.logger
        self.qtable = qtable
        self.chip = chip
        self.low_prec = low_prec
        self.high_prec = high_prec
        self.cali_table_name = cali_table_name
        self.parser = parser

    def _num_units(self, state: SearchState) -> int:
        return len(self.ops)

    def _propose_unit(self, idx: int, state: SearchState) -> Optional[Trial]:
        from .mix_precision import MixQuantModel
        op = self.ops[idx]
        layer_type = self.parser.get_op_type_by_op_name(op)
        self.logger.print_info("start to handle layer: {}, type: {}".format(op, layer_type))
        self.fp_layer_list.remove(op)
        mix_table = self.mix_prec._gen_mix_table(self.fp_layer_list, self.qtable)
        mix_model = MixQuantModel(state.fp32_mlir,
                                  self.chip,
                                  self.low_prec,
                                  self.high_prec,
                                  self.cali_table_name,
                                  mix_table,
                                  using_cuda=True)
        return Trial(op=op, mix_model=mix_model, mix_table=mix_table)

    def _commit_unit(self, trial: Trial, idx: int, state: SearchState) -> None:
        pass

    def _rollback_unit(self, trial: Trial, idx: int, state: SearchState) -> None:
        self._restore_op(trial)

    def advance(self, trial: Trial, state: SearchState) -> None:
        self._restore_op(trial)
        self._cursor += 1

    def _restore_op(self, trial: Trial) -> None:
        if trial.op not in self.fp_layer_list:
            self.fp_layer_list.append(trial.op)


class ClusterAcceptanceTry(CursorTryStrategy):
    """Cluster-acceptance loop for 4bit search (Part 3).

    Each unit is a cluster of ops. On KEEP, the cluster stays demoted
    (removed from keep_int8_layers). On ENOUGH (cluster cos below threshold),
    the cluster is rolled back and, if try_cluster_one_by_one is set, a
    nested sub-search tries each op individually before stopping.
    """

    def __init__(self,
                 state: SearchState,
                 sorted_clusters: List[List[str]],
                 keep_int8_layers: List[str],
                 strategy,
                 mix_prec,
                 qtable,
                 classifier,
                 parser,
                 loss_dict,
                 target_cos_threshold: float,
                 retry_eval_factory=None):
        super().__init__(state)
        self.clusters = sorted_clusters
        self.keep_int8_layers = keep_int8_layers
        self.strategy = strategy
        self.mix_prec = mix_prec
        self.logger = mix_prec.logger
        self.qtable = qtable
        self.classifier = classifier
        self.parser = parser
        self.loss_dict = loss_dict
        self.try_cluster_one_by_one = True
        self.accepted_outputs_cos = state.baseline_cos \
            if state.baseline_cos is not None else target_cos_threshold
        # Factory for the one-by-one retry sub-eval — produces an eval with
        # the SAME metric (cos or end2end) as the parent cluster loop, but
        # use_enough=False (retry tries all ops). When None, falls back to
        # a plain cos eval.
        self.retry_eval_factory = retry_eval_factory

    def _num_units(self, state: SearchState) -> int:
        return len(self.clusters)

    def _propose_unit(self, idx: int, state: SearchState) -> Optional[Trial]:
        cluster = self.clusters[idx]
        for op_name in cluster:
            self.strategy.before_trial(state.searcher, op_name)
            self.keep_int8_layers.remove(op_name)
        high_prec_type = self.strategy.qtable_keep_mode(state.searcher)
        mix_table = self.mix_prec._gen_mix_table(mix_ops=self.keep_int8_layers,
                                                 qtable=self.qtable,
                                                 high_prec_type=high_prec_type)
        mix_model = self.strategy.make_trial_model(state.searcher, mix_table)
        return Trial(op=cluster[0] if len(cluster) == 1 else None,
                     cluster=cluster,
                     mix_model=mix_model,
                     mix_table=mix_table)

    def _commit_unit(self, trial: Trial, idx: int, state: SearchState) -> None:
        self.accepted_outputs_cos = trial.score

    def _rollback_unit(self, trial: Trial, idx: int, state: SearchState) -> None:
        for op_name in trial.cluster:
            self.strategy.rollback_trial(state.searcher, op_name)
            self.keep_int8_layers.append(op_name)

    def advance(self, trial: Trial, state: SearchState) -> None:
        if trial.verdict == Verdict.ENOUGH and self.try_cluster_one_by_one:
            self._run_one_by_one_retry(trial.cluster, state)
        self._cursor += 1

    def _sort_cluster_by_loss(self, cluster):
        return sorted(cluster, key=lambda op_name: self.loss_dict[op_name][0])

    def _run_one_by_one_retry(self, cluster: List[str], state: SearchState) -> None:
        sorted_cluster = self._sort_cluster_by_loss(cluster)
        self.logger.print_info(f'Cluster {cluster} failed, try ops one by one: {sorted_cluster}')
        sub_try = LowPrecOneByOneTry(state=state,
                                     ops=sorted_cluster,
                                     keep_int8_layers=self.keep_int8_layers,
                                     strategy=self.strategy,
                                     mix_prec=self.mix_prec,
                                     qtable=self.qtable,
                                     parser=self.parser,
                                     init_score=self.accepted_outputs_cos)
        # Use the SAME metric as the parent cluster loop (cos or end2end),
        # with use_enough=False so the retry tries every op instead of
        # stopping on the first failure. retry_eval_factory is provided by
        # the searcher (SearchQtable4Bit.run) via _make_eval; falls back to
        # a plain cos eval when not provided.
        if self.retry_eval_factory is not None:
            sub_eval = self.retry_eval_factory()
        else:
            from .cali_eval import CosEval
            sub_eval = CosEval(target=state.target_cos_threshold,
                               direction='higher_better',
                               mix_prec=self.mix_prec,
                               global_compare_layers=state.ctx.get('gcl', []),
                               layers_rate=state.ctx.get('lr', []),
                               predictions_gt=state.ctx.get('gt', []),
                               expected=state.expected_cos,
                               num_sample=state.num_sample,
                               use_enough=False,
                               target_semantics='floor',
                               desc='one_by_one')
        SearchLoop(sub_try, sub_eval, desc='one_by_one', show_pbar=False).run(state)
        self.accepted_outputs_cos = sub_try.accepted_outputs_cos


class LowPrecOneByOneTry(CursorTryStrategy):
    """One-by-one retry for ops in a failed cluster (Part 3 nested sub-search).

    Demotes each op individually; KEEP = cos >= threshold (op stays demoted),
    WITHDRAW = cos < threshold (op restored). No ENOUGH — tries all ops.
    """

    def __init__(self,
                 state: SearchState,
                 ops: List[str],
                 keep_int8_layers: List[str],
                 strategy,
                 mix_prec,
                 qtable,
                 parser,
                 init_score: Optional[float] = None):
        super().__init__(state)
        self.ops = ops
        self.keep_int8_layers = keep_int8_layers
        self.strategy = strategy
        self.mix_prec = mix_prec
        self.qtable = qtable
        self.parser = parser
        self.accepted_outputs_cos = init_score \
            if init_score is not None else state.target_cos_threshold

    def _num_units(self, state: SearchState) -> int:
        return len(self.ops)

    def _propose_unit(self, idx: int, state: SearchState) -> Optional[Trial]:
        op = self.ops[idx]
        self.keep_int8_layers.remove(op)
        self.strategy.before_trial(state.searcher, op)
        high_prec_type = self.strategy.qtable_keep_mode(state.searcher)
        mix_table = self.mix_prec._gen_mix_table(mix_ops=self.keep_int8_layers,
                                                 qtable=self.qtable,
                                                 high_prec_type=high_prec_type)
        mix_model = self.strategy.make_trial_model(state.searcher, mix_table)
        return Trial(op=op, cluster=[op], mix_model=mix_model, mix_table=mix_table)

    def _commit_unit(self, trial: Trial, idx: int, state: SearchState) -> None:
        self.accepted_outputs_cos = trial.score

    def _rollback_unit(self, trial: Trial, idx: int, state: SearchState) -> None:
        self.strategy.rollback_trial(state.searcher, trial.op)
        self.keep_int8_layers.append(trial.op)
