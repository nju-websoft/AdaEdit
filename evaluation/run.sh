#!/bin/bash

# Variables
## sft models
ck_path="$1"
eval_dir="$2"
edit_format="$3"
if [ -z "$4" ]; then
    output_type=''
else
    output_type="$4"
fi

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
        --ck_path $ck_path \
        --edit_format $edit_format \
        $optional_arg \
        --parallel_size $parallel_size
}

# generation for supported edit benchmarks
conda activate AdaEdit
call_main "edit" 4 $output_type
conda deactivate

# evaluation
# HumanEvalFix
conda activate humanevalfix
call_main "humanevalfix-python" 0 $output_type
# call_main "humanevalfix-js" 0 $output_type
conda deactivate

#  EditEval
conda activate AdaEdit
call_main "editeval" 0 $output_type
conda deactivate

# CanItEdit
conda activate canitedit
call_main "canitedit" 0 $output_type
conda deactivate
