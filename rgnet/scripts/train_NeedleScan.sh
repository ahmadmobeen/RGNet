#!/bin/bash
source /home/mobeen/codes/NeedleScan/.venv/bin/activate
cd $(dirname $0)/../..
######## Data paths, representing the federated NeedleScan datasets
train_path=/gpfs/public/datasets/vmr/rgnet/unified/train.jsonl
eval_path=/gpfs/public/datasets/vmr/rgnet/unified/val.jsonl
eval_split_name=val
results_root=/gpfs/public/artifacts/vmr/rgnet

######## Setup video/textual feature path
motion_feat_dir=/gpfs/public/datasets/vmr/rgnet/unified/features/video_features_768.lmdb
appearance_feat_dir=/gpfs/public/datasets/vmr/rgnet/unified/features/video_features_768.lmdb
text_feat_dir=/gpfs/public/datasets/vmr/rgnet/unified/features/text_features_768.lmdb

# Feature dimension (SigLIP base=768, CLIP ViT-B/32 text=512)
v_motion_feat_dim=768
v_appear_feat_dim=768
t_feat_dim=768

#### training
n_epoch=35
lr=1e-4
lr_drop=25
device_id=0
num_queries=5
max_v_l=90  # 90 segments * 2.0s = 180s context window
bsz=8
eval_bsz=8
clip_length=2.0  ##  video features extracted every 2.0 seconds via VideoEncoder
max_q_l=25
num_workers=4

######## Hyper-parameter
dset_name=unified
seed=2020
adapter_module=none
max_es_cnt=-1
eval_epoch_interval=5
topk_window=30
start_epoch_for_adapter=100
adapter_loss_coef=0.2
retrieval_loss_coef=10
pos_temperature=100
exp_id=train_needlescan

# Using standard torchrun instead of srun wrapper
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 PYTHONPATH=$PYTHONPATH:. torchrun --nproc_per_node=2 --master_port $RANDOM rgnet/train.py \
--gumbel_eps 0.3 \
--gumbel_single_proj \
--nms_thd 0.5 \
--seed ${seed} \
--clip_length ${clip_length}  \
--max_es_cnt ${max_es_cnt} \
--topk_window ${topk_window} \
--eval_epoch_interval ${eval_epoch_interval} \
--start_epoch_for_adapter ${start_epoch_for_adapter} \
--lr ${lr} \
--lr_drop ${lr_drop} \
--n_epoch ${n_epoch} \
--max_v_l ${max_v_l} \
--max_q_l ${max_q_l} \
--dset_name ${dset_name} \
--train_path ${train_path} \
--eval_split_name ${eval_split_name} \
--motion_feat_dir ${motion_feat_dir} \
--appearance_feat_dir ${appearance_feat_dir} \
--t_feat_dir ${text_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--v_appear_feat_dim ${v_appear_feat_dim} \
--v_motion_feat_dim ${v_motion_feat_dim} \
--bsz ${bsz} \
--eval_bsz ${eval_bsz} \
--results_root ${results_root} \
--exp_id ${exp_id} \
--num_queries ${num_queries} \
--num_workers ${num_workers} \
--adapter_module ${adapter_module} \
--adapter_loss_coef ${adapter_loss_coef} \
--qddetr \
--retrieval_loss_coef ${retrieval_loss_coef} \
--enc_layers 2 \
--dec_layers 6 \
--dec_layers_2 2 \
--no_adapter_loss \
--winret \
--no_neg_contrast_loss \
--ret_eval \
--resume_all \
--gumbel \
--gumbel_2 \
--gumbel_3 \
--dabdetr \
--pos_temperature ${pos_temperature} \
--dim_feedforward 2048 \
${@:1}
