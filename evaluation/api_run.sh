#!/bin/bash

# Usage:
#   bash api_run.sh <api_model_key> <eval_dir> <edit_format> [output_type]

# API environment variables:
# export OPENAI_API_KEY="your_api_key"
# export OPENAI_BASE_URL="https://your-openai-compatible-endpoint/v1"

api_model_key="$1"
eval_dir="$2"
edit_format="$3"
if [ -z "$4" ]; then
    output_type=''
else
    output_type="$4"
fi

# Force one-sample deterministic decoding for supported benchmarks.
temperature=0
n_samples=1

function call_main() {
    local datasets=$1
    local parallel_size=$2
    local output_type=$3

    local optional_arg=""
    if [ -n "$output_type" ]; then
        optional_arg="--output_type $output_type"
    fi

    python main.py \
        --save_dir $eval_dir \
        --datasets $datasets \
        --edit_format $edit_format \
        --backend api \
        --api_model_key $api_model_key \
        --temperature $temperature \
        --n_samples $n_samples \
        --report 1 \
        $optional_arg \
        --parallel_size $parallel_size
}

# generation for supported edit benchmarks
conda activate AdaEdit
call_main "edit" 1 $output_type
# call_main "humanevalfix-js" 1 $output_type
conda deactivate

# evaluation
# HumanEvalFix
conda activate humanevalfix
call_main "humanevalfix-python" 0 $output_type
# call_main "humanevalfix-js" 0 $output_type
conda deactivate

# EditEval
conda activate AdaEdit
call_main "editeval" 0 $output_type
conda deactivate

# CanItEdit
conda activate canitedit
call_main "canitedit" 0 $output_type
conda deactivate
