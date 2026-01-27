#!/bin/bash

MODEL=$1
GPU_ID=1
OUTPUT_PATH=$2


echo $MODEL
echo $GPU_ID
echo $OUTPUT_PATH

export CUDA_VISIBLE_DEVICES=$GPU_ID

source $(conda info --base)/etc/profile.d/conda.sh
mkdir -p $OUTPUT_PATH

conda activate lmeval

lm_eval --model hf \
    --model_args pretrained=$MODEL \
    --tasks ifeval \
    --device cuda:$GPU_ID \
    --batch_size 8 \
    --output_path $OUTPUT_PATH

lm_eval --model hf \
    --model_args pretrained=$MODEL \
    --tasks mmlu_stem,lambada_openai \
    --device cuda:$GPU_ID \
    --batch_size 16 \
    --output_path $OUTPUT_PATH

conda deactivate
conda activate bigcode
cd bigcode-evaluation-harness

accelerate launch  main.py \
  --model $MODEL \
  --max_length_generation 512 \
  --precision bf16 \
  --tasks conala \
  --temperature 0.2 \
  --n_samples 10 \
  --batch_size 32 \
  --allow_code_execution \
  --metric_output_path $OUTPUT_PATH/code_val.json \
  --use_auth_token

cd ..
conda deactivate
conda activate safety-eval
cd safety-eval-fork

export OPENAI_API_KEY=''

python evaluation/eval.py generators \
  --model_name_or_path $MODEL \
  --model_input_template_path_or_name llama3 \
  --tasks wildjailbreak:harmful \
  --report_output_path $OUTPUT_PATH/safety_val.json \
  --save_individual_results_path $OUTPUT_PATH/safety_generation.json \
  --batch_size 16
  
