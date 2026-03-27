#!/bin/bash
source /home/mobeen/codes/NeedleScan/.venv/bin/activate
cd /home/mobeen/codes/NeedleScan/research/RGNet

export WANDB_BASE_URL="https://pyler.wandb.io"
echo "Running RGNet Evaluation on Unified Validation Set..."
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=$PYTHONPATH:. python rgnet/inference.py \
--resume /gpfs/public/artifacts/vmr/rgnet/unified-train_needlescan/model_e0032.ckpt \
--eval_split_name val \
--eval_path /gpfs/public/datasets/vmr/rgnet/unified/val.jsonl \
--eval_id unified_eval_final \
--num_workers 8 \
--bsz 128 \
--ret_eval \
--topk_window 40 \
--nms_thd 0.5 \
--topk 10 \
--topk_span 5 \
--max_after_nms 100
