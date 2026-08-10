#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
"""Reference: end-to-end top1/top5 eval on a MixQuantModel.

Doubles as (a) a test for ``End2EndEval`` — A/B vs
``model_eval.py --postprocess_type topx`` on the same model+dataset should
produce identical top1/top5 — and (b) the customer template for plugging in
a custom metric (mAP, perplexity, WER, ...).

Usage (A/B vs model_eval.py):

  # Path A (file-based, current production):
  model_eval.py --model_file resnet18.mlir --postprocess_type topx \\
      --dataset /data/imagenet/val --label_file imagenet_labels.txt \\
      --count 500

  # Path B (in-memory, via MixModelInference — this file):
  python python/calibration/end2end_topx_example.py \\
      --mlir resnet18.mlir --chip bm1684x \\
      --dataset /data/imagenet/val \\
      --label_file imagenet_labels.txt --count 500

  Assert: top1_A == top1_B and top5_A == top5_B.

Customer template steps to plug a custom metric:

  1. Implement a ``base_class`` subclass in
     ``python/eval/postprocess_and_score_calc/<your_metric>.py`` with
     ``init / preproc / update / get_result / print_info / reset``
     (see ``topx.py`` for the shape; ``reset`` clears per-trial counters
     while keeping GT parsed in ``init``).

  2. Point ``--postprocess_type`` at your new plugin name.

  3. (For search-loop integration) write an ``eval_config.json`` with
     ``"eval_metric": "<your_metric>"`` and ``"score_plugin":
     "<your_metric>"``, then pass to ``run_calibration.py --eval_config
     eval_config.json``.

See ``docs/search-qtable-eval-merge-supplement.html`` for the design and
``docs/search-qtable-unified-eval-proposal.html`` §9.3-9.4 for the eval
strategy hierarchy.
"""
import argparse
import types

from calibration.mix_precision import MixQuantModel
from calibration.cali_eval import MixModelInference


def _build_args(mlir, chip, dataset, label_file, count, postprocess_type):
    """Construct a minimal args Namespace compatible with common_inference.

    ``common_inference.__init__`` and ``base_class.init`` read these fields.
    """
    return types.SimpleNamespace(
        model_file=mlir,
        chip=chip,
        postprocess_type=postprocess_type,
        dataset=dataset,
        data_list='',
        label_file=label_file,
        count=count,
        debug_cmd={},
        batch_size=1,
        net_input_dims=[],
        cuda=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--mlir', required=True, help='mlir file path')
    parser.add_argument('--chip', required=True, help='chip name (e.g. bm1684x)')
    parser.add_argument('--dataset',
                        required=True,
                        help='dataset root directory (e.g. /data/imagenet/val)')
    parser.add_argument('--label_file',
                        required=True,
                        help='label file (same format as model_eval.py)')
    parser.add_argument('--count',
                        type=int,
                        default=0,
                        help='number of samples (0 = all in dataset)')
    parser.add_argument('--postprocess_type',
                        default='topx',
                        help='score plugin name (topx | coco_mAP | <custom>)')
    args = parser.parse_args()

    # Build the in-memory model (search-loop style). mode/high_prec=None
    # gives the FP32 model; for a quantized eval set mode/high_prec to the
    # quantized mode (e.g. 'INT8' for low_prec, 'F16' for high_prec).
    mix_model = MixQuantModel(args.mlir, args.chip, None, 'F32')

    # Wire the model_eval pipeline. img_proc + score plugin (topx) are set
    # up once here; per-trial swap (in the search loop) would call
    # runner.set_model(...) + runner.reset_score().
    runner = MixModelInference(
        _build_args(args.mlir, args.chip, args.dataset, args.label_file, args.count,
                    args.postprocess_type), mix_model)

    # Drive common_inference.run(idx, img_path, label) over the sample list.
    # Batching by batch_size is handled inside common_inference.run; the
    # remainder is flushed by get_result().
    n = args.count if args.count > 0 else len(runner._samples)
    for idx in range(n):
        img_path, label = runner.get_sample(idx)
        if img_path is None:
            break
        runner.run(idx, img_path, label)

    result = runner.get_result()
    if isinstance(result, tuple):
        top1, top5 = result
        print('top1={:.4f}  top5={:.4f}  (n={})'.format(top1, top5, runner.score.idx))
    else:
        print('result={}  (n={})'.format(result, runner.score.idx))


if __name__ == '__main__':
    main()
