# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import abc


class base_class(metaclass=abc.ABCMeta):

    # Data-path capability. Default 'toolkit': the harness owns dataset
    # selection, preprocessing (the mlir's img_proc), and sample enumeration;
    # the plugin only accumulates via update(). Set to 'plugin' to own the
    # dataset + enumeration + preprocessing yourself (tier-2): then you MUST
    # implement __iter__ (yield your samples) and preproc(sample) -> model
    # input tensor, and the harness drives inference over YOUR samples. For
    # full self-sufficiency incl. inference (tier-3), leave data_mode as-is
    # and return 'self' from begin_model instead.
    data_mode = 'toolkit'

    def __init__(self, args):
        self._init_args = args
        self.init(args)

    @abc.abstractmethod
    def init(self, args):
        pass

    def reset(self):
        """Clear per-trial accumulators without re-running ``init()``.

        Called by ``MixModelInference`` between trials in the search loop:
        the model changes per trial, but GT parsed in ``init()`` (label
        file, COCO json, etc.) is reused. Default fallback rebuilds via
        ``init()``; plugins with expensive init() override to clear only
        the per-trial counters and keep the parsed GT.
        """
        self.init(self._init_args)

    # ------------------------------------------------------------------
    # OPTIONAL model-aware hooks (default no-ops -> current behavior).
    #
    # These let a plugin learn WHICH model it is scoring and, if it wants,
    # take over the whole eval (deploy to bmodel / download to a board /
    # run a remote session). Called once per trial by the search harness:
    #
    #     run_eval_trial(): reset() -> begin_model(ctx) -> [drive|self]
    #                       -> get_result() -> end_model()
    #
    # ``model_ctx`` is a SimpleNamespace the harness fills with the trial's
    # model artifacts, e.g.:
    #     chip, fp32_mlir, quanted_mlir_file, weight_file, cali_table,
    #     mix_table, using_cuda, n_fp (float-layer count), phase
    # so the plugin can compile / deploy the exact model under test.
    # ------------------------------------------------------------------

    def begin_model(self, model_ctx):
        """Per-trial model preparation. Return value selects the eval mode:

        - ``None`` (default): the harness drives sample enumeration +
          inference; the plugin only accumulates via ``update()``.
        - ``'self'``: the plugin runs its OWN full eval loop here (its own
          dataset, preprocess, inference — e.g. a remote board session) and
          accumulates internally; the harness then just calls
          ``get_result()``. ``begin_model`` runs AFTER ``reset()``, so any
          accumulation done here survives to ``get_result()``.

        Determinism rule: whatever samples you enumerate must be the SAME,
        in the SAME order, on every trial -- otherwise the search's
        float-layer comparisons become noise.
        """
        return None

    def end_model(self):
        """Per-trial teardown (kill remote session, free device handles).
        Called after ``get_result()``. Default no-op."""
        pass

    # ------------------------------------------------------------------
    # OPTIONAL customer-owned data path (tier-2, data_mode == 'plugin').
    #
    # When data_mode == 'plugin' the harness stops using the mlir's img_proc
    # and its ImageFolder/DataSelector sample list; instead it drives:
    #
    #     open_dataset(args)          # once per search (optional)
    #     for sample in self:         # __iter__ (REQUIRED), same order every trial
    #         x = self.preproc(sample)         # -> model input tensor (or list)
    #         outputs = <harness inference>    # simulator invoke
    #         self.update(idx, outputs, sample)
    #
    # This lets a customer use their own dataset format (audio, text, paired /
    # multi-input, custom containers), own preprocessing, and own postprocess
    # in update(), while still reusing the harness's simulator inference. For
    # own inference too (remote board), use begin_model -> 'self' instead.
    #
    # Determinism rule: __iter__ MUST yield the SAME samples in the SAME order
    # on every trial (shuffle once in open_dataset/init, then freeze), or the
    # search's float-layer comparisons become noise.
    # ------------------------------------------------------------------

    def open_dataset(self, args):
        """Optional one-time dataset preparation (load index/annotations,
        build the sample list). Called once before the first trial when
        ``data_mode == 'plugin'``. Default no-op — do it in ``init`` if you
        prefer. ``__iter__`` is what actually yields samples per trial."""
        pass

    # ------------------------------------------------------------------
    # OPTIONAL all-in-one per-sample hook (preprocess + inference +
    # postprocess). This is where the customer decides WHERE those three
    # run: locally, or shipped to a board / remote session (prepared in
    # begin_model). Default returns None -> the harness uses the built-in
    # path (the mlir's img_proc + the simulator) and hands raw outputs to
    # update().
    #
    # The return value ("predictions") is OPAQUE to the harness: it is
    # produced here and consumed by THIS plugin's update(), so it can be
    # anything (raw outputs, decoded boxes, a class id, a count, ...). The
    # only value the harness interprets is get_result() -> float/tuple.
    #
    # Ground truth stays on the host (parsed in init, kept on self); the
    # accuracy comparison happens on the host in update(). So only the
    # ``sample`` goes out and the compact ``predictions`` come back.
    # ------------------------------------------------------------------

    def eval_sample(self, sample, preprocess):
        """Preprocess + run + postprocess ONE sample; return predictions.

        - ``sample``: the harness img_path (tier-1) or your own object
          (tier-2 ``__iter__``).
        - ``preprocess``: the harness's default (mlir img_proc) callable --
          call ``preprocess(sample)`` to reuse the common preprocessing, or
          ignore it and do your own.
        - Return the predictions (opaque; consumed by your ``update``), or
          ``None`` to fall back to the built-in img_proc + simulator path.
        """
        return None

    @abc.abstractmethod
    def preproc(self, img_paths):
        pass

    @abc.abstractmethod
    def update(self, idx, outputs, img_paths=None, labels=None, ratios=None):
        pass

    @abc.abstractmethod
    def get_result(self):
        pass

    @abc.abstractmethod
    def print_info(self):
        pass
