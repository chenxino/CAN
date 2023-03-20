#!/bin/bash

gpu_n=0
DATASET="SWaT_10"
random_seed=0
group='1-1'
normalize='True'
val_split=0.1
shuffle_dataset="True"
print_every=1
log_tensorboard="False"
debug="False"

# --- Predictor params ---
scale_scores="True"  # 是否在get_score中使用中值和iqr进行正规化
use_mov_av="False" # 是否使用滑动窗口平均
gamma=0 # 重构分数占比
# level=None
# q=None
dynamic_pot="False" # POT_eval中的一个参数
topk=2

# --- model params ---
seq_len=5
d_model=8
n_heads=8
e_layers=3
d_ff=16
factor=5
padding=0
dropout=0.2
attn="full"
activation='gelu'
use_graph="True"
k=0
use_AE="AE"
start_len=0
subgraph_size=10
e_dim=10

# --- train params ---
train_epochs=20
batch_size=16
patience=5
learning_rate=0.0001
use_gpu="True"
run_mode="all"
# run_mode="fore"
des="all"

# 读入参数
gpu_n=$1
DATASET=$2



if [[ "$gpu_n" == "cpu" ]]; then
    python train.py --dataset $DATASET --random_seed $random_seed --group $group --normalize $normalize --val_split $val_split --shuffle_dataset $shuffle_dataset --print_every $print_every --log_tensorboard $log_tensorboard --debug $debug\
    --scale_scores $scale_scores --use_mov_av $use_mov_av --gamma $gamma --dynamic_pot $dynamic_pot --topk $topk\
    --seq_len $seq_len --d_model $d_model --n_heads $n_heads --e_layers $e_layers --d_ff $d_ff --factor $factor --padding $padding --dropout $dropout --attn $attn --activation $activation --use_graph $use_graph --k $k --use_AE $use_AE --start_len $start_len --subgraph_size $subgraph_size\
    --train_epochs $train_epochs --batch_size $batch_size --patience $patience --learning_rate $learning_rate --use_gpu $use_gpu --run_mode $run_mode --des $des 

else
    CUDA_VISIBLE_DEVICES=$gpu_n python train.py --dataset $DATASET --random_seed $random_seed --group $group --normalize $normalize --val_split $val_split --shuffle_dataset $shuffle_dataset --print_every $print_every --log_tensorboard $log_tensorboard --debug $debug\
    --scale_scores $scale_scores --use_mov_av $use_mov_av --gamma $gamma --dynamic_pot $dynamic_pot --topk $topk\
    --seq_len $seq_len --d_model $d_model --n_heads $n_heads --e_layers $e_layers --d_ff $d_ff --factor $factor --padding $padding --dropout $dropout --attn $attn --activation $activation --use_graph $use_graph --k $k --use_AE $use_AE --start_len $start_len --subgraph_size $subgraph_size\
    --train_epochs $train_epochs --batch_size $batch_size --patience $patience --learning_rate $learning_rate --use_gpu $use_gpu --run_mode $run_mode --des $des --e_dim $e_dim
fi
# e_layers=(3 4 5 6)
# for i in ${e_layers[*]}; do
# CUDA_VISIBLE_DEVICES='1' python new_train.py --dataset SWaT_10 --n_heads 8 --d_model 8 --d_ff 16 --batch_size 16 --train_epochs 5 --seq_len 5 --e_layers $i --use_graph True --des base_line;
# done