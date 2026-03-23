#!/bin/bash
source /home/mobeen/codes/NeedleScan/.venv/bin/activate
cd /home/mobeen/codes/NeedleScan/research/RGNet

echo "Running RGNet Evaluation on Unified Validation Set..."
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=$PYTHONPATH:. python rgnet/inference.py \
--resume results_needlescan/unified-short_test_needlescan/model_e0000.ckpt \
--eval_split_name val \
--eval_path data/unified/val_short.jsonl \
--eval_id short_test_eval \
--num_workers 8 \
--bsz 32 \
--ret_eval \
--topk_window 40 \
--nms_thd 0.5 \
--topk 10 \
--topk_span 5 \
--max_after_nms 100
