#!/usr/bin/env


# Number of GPUs per GPU worker
GPUS_PER_NODE=4
# Number of GPU workers, for single-worker training, please set to 1
WORKER_CNT=1
# The ip address of the rank-0 worker, for single-worker training, please set to localhost
export MASTER_ADDR='localhost'
# The port for communication
export MASTER_PORT=8401
# The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
export RANK=0

export PYTHONPATH=${PYTHONPATH}:`pwd`/src/

#DATAPATH=/phd/SPU/spu_pretrain

dataset=Office

# data options
train_data=./amazon_2018/finetune_data/${dataset}/
emb_path=./amazon_2018/finetune_data/${dataset}_baichuan_lmdb
input_dim=4096
output_dim=4096
train_stage=finetune


test_user_data=./amazon_2018/finetune_data/${dataset}/
test_item_data=./amazon_2018/finetune_data/${dataset}/smap.json
test_emb_lmdb=./amazon_2018/finetune_data/${dataset}_baichuan_lmdb


resume=./exp/pretrain_u2iLoss_pred-next-item_ep30_warmup1/checkpoints/epoch20.pt

# output options
output_base_dir=./exp/
name=ft_${dataset}_ep20_lr1e-5_maxEp200_pretrain_u2iLoss_pred-next-item_ep30
save_step_frequency=50000 # disable it
save_epoch_frequency=1
log_interval=10
report_training_batch_acc="--report-training-batch-acc"

# training hyper-params
max_one_year_length=40
max_one_month_length=1
sample_one_year_length=4
sample_one_month_length=1
num_workers=8
warmup=0
batch_size=16
accum_freq=1
lr=2e-5 # 1e-5 for Scientific
wd=0.0
max_epochs=200
valid_step_interval=10000000  # not valid
valid_epoch_interval=100  # not valid
precision='amp'
wandb_project='LEARN_amazon2018'
num_layers=4


torchrun --nproc_per_node=${GPUS_PER_NODE} --nnodes=${WORKER_CNT} --node_rank=${RANK} \
        --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} src/training/main_amazon.py \
        --train-data=${train_data} \
        --emb-path=${emb_path} \
        --num-workers=${num_workers} \
        --num-layers=${num_layers} \
        --wandb_project=${wandb_project} \
        --resume=${resume} \
        --log_dir=${output_base_dir} \
        --name=${name} \
        --save-step-frequency=${save_step_frequency} \
        --save-epoch-frequency=${save_epoch_frequency} \
        --log-interval=${log_interval} \
        ${report_training_batch_acc} \
        --max-one-year-length=${max_one_year_length} \
        --max-one-month-length=${max_one_month_length} \
        --sample-one-year-length=${sample_one_year_length} \
        --sample-one-month-length=${sample_one_month_length} \
        --warmup=${warmup} \
        --batch-size=${batch_size} \
        --valid-step-interval=${valid_step_interval} \
        --valid-epoch-interval=${valid_epoch_interval} \
        --accum-freq=${accum_freq} \
        --lr=${lr} \
        --wd=${wd} \
        --max-epochs=${max_epochs} \
        --model-name=${model_name} \
        --precision=${precision} \
        --input-dim=${input_dim} \
        --output-dim=${output_dim} \
        --train-stage=${train_stage} \
        --ft-data=${dataset} \
        --test-user-data=${test_user_data} \
        --test-item-data=${test_item_data} \
        --test-emb-lmdb=${test_emb_lmdb} \
        --wandb_project=${wandb_project} \



