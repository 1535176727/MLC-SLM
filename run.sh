#!/bin/bash
. ./path.sh || exit 1;

#export TORCH_DISTRIBUTED_TIMEOUT=1800
#export NCCL_ASYNC_ERROR_HANDLING=1
#export NCCL_BLOCKING_WAIT=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_LAUNCH_TIMEOUT=6000
#export NCCL_IB_DISABLE=1
export NCCL_TIMEOUT=6000
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=6000
unset LD_LIBRARY_PATH
export LD_LIBRARY_PATH=~/my_libs:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES="4,5,6,7"
echo "CUDA_VISIBLE_DEVICES is ${CUDA_VISIBLE_DEVICES}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

stage=4
stop_stage=5

# You should change the following two parameters for multiple machine training,
# see https://pytorch.org/docs/stable/elastic/run.html
HOST_NODE_ADDR="localhost:0"
num_nodes=1
job_id=2023
nj=10

# data_type can be `raw` or `shard`. Typically, raw is used for small dataset,
# `shard` is used for large dataset which is over 1k hours, and `shard` is
# faster on reading data and training.
data_type=raw
num_utts_per_shard=1000

train_data_dir=train_data
train_segments_path=train_data/segments
train_split_dir=train_data/split
train_split_log_dir=train_data/split_log
dev_data_dir=dev_data
test_data_dir=test_data_mlc
dev_segments_path=dev_data/segments
dev_split_dir=dev_data/split
dev_split_log_dir=dev_data/split_log
cmd=local/run.pl

step1_train_config=conf/train_SHNU-mASR_step1_lora.yaml
#step2_train_config=conf/train_SHNU-mASR_step2.yaml
step3_train_config=conf/train_SHNU-mASR_step3_lora.yaml

step1_dir=SHNU-masr/step1_gated_bicross_attn_with_cat #just for training projector
#step2_dir=SHNU-masr/step2 #training projector and encoder
step3_dir=SHNU-masr/step3_gated_bicross_attn_with_cat_trainable_projector #training projector, encoder and qwen

tensorboard_dir=tensorboard
step1_checkpoint=
step2_checkpoint=
step3_checkpoint=
num_workers=8
prefetch=10

# use average_checkpoint will get better result
average_checkpoint=true
decode_checkpoint=$step3_dir/final.pt
average_num=3
decode_mode=SHNU-mASR_decode

train_engine=deepspeed

deepspeed_config1=conf/ds_stage1.json
deepspeed_config2=conf/ds_stage2.json
deepspeed_config3=conf/ds_stage3.json
deepspeed_save_states=""model+optimizer""

. tools/parse_options.sh || exit 1;


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "Stage 0: Generate segments"

    ln -s your_training_data_path ./train_data
    ln -s your_dev_data_path ./dev_data

    python local/prepare_segments.py --data_dir $train_data_dir --segments_path $train_segments_path
    log "Segments file of training dataset is saved into $train_segments_path"

    python local/prepare_segments.py --data_dir $dev_data_dir --segments_path $dev_segments_path
    log "Segments file of development dataset is saved into $dev_segments_path"
    # About 1min
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  log "Stage 1: Split wavs using segments"
    
  log "Split train dataset wavs"
  nutt=$(<${train_segments_path} wc -l)
  nj=$((nj<nutt?nj:nutt))
  mkdir -p $train_split_log_dir
  split_segments=""
  for n in $(seq ${nj}); do
      split_segments="${split_segments} ${train_split_log_dir}/segments.${n}"
  done
  local/split_scp.pl "${train_segments_path}" ${split_segments}

  # shellcheck disable=SC2046
  ${cmd} "JOB=1:${nj}" "${train_split_log_dir}/split_wavs.JOB.log" \
      python local/split_wav.py \
          "--segments_path=${train_split_log_dir}/segments.JOB" \
          "--output_dir=${train_split_dir}/split.JOB"

  cat ${train_split_dir}/split.*/wav.scp | shuf > $train_data_dir/wav.scp
  cat ${train_split_dir}/split.*/text | shuf > $train_data_dir/text

  log "Split dev dataset wavs"
  nutt=$(<${dev_segments_path} wc -l)
  nj=$((nj<nutt?nj:nutt))
  mkdir -p $dev_split_log_dir
  split_segments=""
  for n in $(seq ${nj}); do
      split_segments="${split_segments} ${dev_split_log_dir}/segments.${n}"
  done
  local/split_scp.pl "${dev_segments_path}" ${split_segments}

  # shellcheck disable=SC2046
  ${cmd} "JOB=1:${nj}" "${dev_split_log_dir}/split_wavs.JOB.log" \
      python local/split_wav.py \
          "--segments_path=${dev_split_log_dir}/segments.JOB" \
          "--output_dir=${dev_split_dir}/split.JOB"

  cat ${dev_split_dir}/split.*/wav.scp | shuf > $dev_data_dir/wav.scp
  cat ${dev_split_dir}/split.*/text | shuf > $dev_data_dir/text
  
  log "Split wavs done"
  # about 21h 10min
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Simple text normalization for the text file"
    
    python local/text_normalization.py \
        --input $train_data_dir/text \
        --output $train_data_dir/text_tn

    python local/text_normalization.py \
        --input $dev_data_dir/text \
        --output $dev_data_dir/text_tn

    log "Text normalization done"
    # about 1min
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  log "Stage 3: Prepare data, prepare required format"
  
  if [ $data_type == "shard" ]; then
    tools/make_shard_list.py --num_utts_per_shard $num_utts_per_shard \
      --num_threads $nj $train_data_dir/wav.scp $train_data_dir/text_tn \
      $(realpath $train_data_dir/shards) $train_data_dir/${data_type}_data.list
    tools/make_shard_list.py --num_utts_per_shard $num_utts_per_shard \
      --num_threads $nj $dev_data_dir/wav.scp $dev_data_dir/text_tn \
      $(realpath $dev_data_dir/shards) $dev_data_dir/${data_type}_data.list
    # about 37min
  else
    tools/make_raw_list.py $train_data_dir/wav.scp $train_data_dir/text_tn \
      $train_data_dir/${data_type}_data_norm.list
    tools/make_raw_list.py $dev_data_dir/wav.scp $dev_data_dir/text_tn \
      $dev_data_dir/${data_type}_data_norm.list
    # about 1min
  fi

  log "Data preparation done"
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  log "Stage 4: Training step 1 start"
  mkdir -p $step1_dir
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  # Use "nccl" if it works, otherwise use "gloo"
  dist_backend="nccl"

  if [ ${train_engine} == "deepspeed" ]; then
    echo "$0: using deepspeed"
  else
    echo "$0: using torch ddp"
  fi

  echo "$0: num_nodes is $num_nodes, proc_per_node is $num_gpus"
  torchrun --nnodes=$num_nodes --nproc_per_node=$num_gpus \
           --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint=$HOST_NODE_ADDR \
    wenet/bin/train_SHNU-mASR.py \
      --train_engine ${train_engine} \
      --config $step1_train_config \
      --data_type  $data_type \
      --train_data $train_data_dir/${data_type}_data_norm.list \
      --cv_data $dev_data_dir/${data_type}_data_norm.list \
      ${step1_checkpoint:+--checkpoint $step1_checkpoint} \
      --model_dir $step1_dir \
      --tensorboard_dir ${tensorboard_dir} \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} \
      --pin_memory \
      --deepspeed_config ${deepspeed_config1} \
      --deepspeed.save_states ${deepspeed_save_states}
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    if [ "$deepspeed_save_states" = "model+optimizer" ]; then
    for subdir in $(find "$step1_dir" -maxdepth 1 -type d | grep -v "^$step1_dir$")
    do
      tag=$(basename "$subdir")
      echo "$tag"
      python3 ${step1_dir}/zero_to_fp32.py \
        ${step1_dir} ${step1_dir}/${tag}.pt -t ${tag}
      rm -rf ${step1_dir}/${tag}
    done
  fi
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  log "Stage 5: Training step 2 start"
  mkdir -p $step2_dir
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  # Use "nccl" if it works, otherwise use "gloo"
  dist_backend="nccl"

  if [ ${train_engine} == "deepspeed" ]; then
    echo "$0: using deepspeed"
  else
    echo "$0: using torch ddp"
  fi

  echo "$0: num_nodes is $num_nodes, proc_per_node is $num_gpus"
  torchrun --nnodes=$num_nodes --nproc_per_node=$num_gpus \
           --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint=$HOST_NODE_ADDR \
    wenet/bin/train_SHNU-mASR.py \
      --train_engine ${train_engine} \
      --config $step2_train_config \
      --data_type  $data_type \
      --train_data $train_data_dir/${data_type}_data_norm.list \
      --cv_data $dev_data_dir/${data_type}_data_norm.list \
      ${step2_checkpoint:+--checkpoint $step2_checkpoint} \
      --model_dir $step2_dir \
      --tensorboard_dir ${tensorboard_dir} \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} \
      --pin_memory \
      --deepspeed_config ${deepspeed_config2} \
      --deepspeed.save_states ${deepspeed_save_states}

fi


if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  log "Stage 5: Training step 3 start"
  mkdir -p $step3_dir
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  # Use "nccl" if it works, otherwise use "gloo"
  dist_backend="nccl"

  if [ ${train_engine} == "deepspeed" ]; then
    echo "$0: using deepspeed"
  else
    echo "$0: using torch ddp"
  fi

  echo "$0: num_nodes is $num_nodes, proc_per_node is $num_gpus"
  torchrun --nnodes=$num_nodes --nproc_per_node=$num_gpus \
           --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint=$HOST_NODE_ADDR \
    wenet/bin/train_SHNU-mASR.py \
      --train_engine ${train_engine} \
      --config $step3_train_config \
      --data_type  $data_type \
      --train_data $train_data_dir/${data_type}_data_norm.list \
      --cv_data $dev_data_dir/${data_type}_data_norm.list \
      ${step3_checkpoint:+--checkpoint $step3_checkpoint} \
      --model_dir $step3_dir \
      --tensorboard_dir ${tensorboard_dir} \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} \
      --pin_memory \
      --deepspeed_config ${deepspeed_config3} \
      --deepspeed.save_states ${deepspeed_save_states}

fi



if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    if [ "$deepspeed_save_states" = "model+optimizer" ]; then
    for subdir in $(find "$step3_dir" -maxdepth 1 -type d | grep -v "^$step3_dir$")
    do
      tag=$(basename "$subdir")
      echo "$tag"
      python3 ${step3_dir}/zero_to_fp32.py \
        ${step3_dir} ${step3_dir}/${tag}.pt -t ${tag}
      rm -rf ${step3_dir}/${tag}
    done
  fi
fi


if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
  # Test model, please specify the model you want to test by --checkpoint
  if [ ${average_checkpoint} == true ]; then
    decode_checkpoint=$step3_dir/avg_${average_num}.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python wenet/bin/average_model.py \
      --dst_model $decode_checkpoint \
      --src_path $step3_dir  \
      --num ${average_num} \
      --val_best
  fi
fi
dev_data_dir=dev_data
test_data_dir=test_data_mlc
test_data_cv=test_data_cv
for test_set in test_data_mlc test_data_cv dev_data; do
results_dir=results/step3_qwen05_cat_trainable_projector_trainable_encoder/$test_set/epoch_1
checkpoint=SHNU-masr/step3_qwen05_cat_trainable_projector_trainable_encoder/epoch_1.pt
step1_dir=SHNU-masr/step3_qwen05_cat_trainable_projector_trainable_encoder
if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
  decode_checkpoint=$step3_dir/avg_${average_num}.pt
  python wenet/bin/recognize_SHNU-mASR.py --gpu 0 \
    --modes $decode_mode \
    --config $step1_dir/train.yaml \
    --data_type $data_type \
    --test_data $test_set/${data_type}_data.list \
    --checkpoint $checkpoint \
    --batch_size 200 \
    --dtype bf16 \
    --result_dir $results_dir
fi
done

for data_set in test_data_mlc test_data_cv dev_data ; do

model_dir=SHNU-mASR/step3_full_fined_whisper_cat_trainable_projector/$data_set/avg_3/SHNU-mASR_decode

if [[ "${data_set}" == "test_data_cvA100" ]] ; then
        langs=(English French German Italian Japanese Korean Portuguese Russian Spanish Thai Vietnamese)
fi

if [[ "${data_set}" == "test_data_mlc" ]] ; then
        langs=(English-American English-Australian English-British English-Filipino English-Indian French German Italian Japanese Korean Portuguese Russian Spanish Thai Vietnamese)
fi

if [[ "${data_set}" == "dev_data" ]] ; then
        langs=(English-American English-Australian English-British English-Filipino English-Indian French German Italian Japanese Korean Portuguese Russian Spanish Thai Vietnamese)
fi
if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ]; then
      python local/text_normalization.py \
        --input $model_dir/text \
        --output $model_dir/text_tn
      #python local/remove_repetition.py --input $result_dir/$decode_mode/text_tn --output $result_dir/$decode_mode/text_tn_rep
      python local/add_space_between_chars_asr.py --input $result_dir/$decode_mode/text --output $result_dir/$decode_mode/text_space
      python local/add_space_between_chars_asr.py --input $test_data_dir/text_tn --output $test_data_dir/text_tn_space

      python local/Whisper_norm.py --input $result_dir/$decode_mode/text_space --output $result_dir/$decode_mode/text_space_norm
      python local/Whisper_norm.py --input $test_data_dir/text_tn_space --output $test_data_dir/text_tn_space_norm

      python local/remove_repetition.py --input $result_dir/$decode_mode/text_space_norm --output $result_dir/$decode_mode/text_space_norm_rep
      #for lang in English French German Japanese Korean; do
#for lang in English-American English-Australian English-British English-Filipino English-Indian French German Italian Japanese Korean Portuguese Russian Spanish Thai Vietnamese; do
#for lang in English French German Italian Japanese Korean Portuguese Russian Spanish Thai Vietnamese; do
for lang in "${langs[@]}"; do
            echo "Processing language: $lang"
          #for lang in Japanese; do
      grep $lang $result_dir/$decode_mode/text_space_norm_rep > $result_dir/$decode_mode/text_space_$lang
      grep $lang $test_data_dir/text_tn_space_norm > $test_data_dir/text_tn_space_$lang
    meeteval-wer wer --partial -r $test_data_dir/text_tn_space_$lang -h $result_dir/$decode_mode/text_space_$lang
  done
  meeteval-wer  wer --partial -r $test_data_dir/text_tn_space_norm -h $result_dir/$decode_mode/text_space_norm_rep
fi
done