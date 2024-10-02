#!/bin/bash
if [ $# -ne 3 ]; then
    echo "Error: run.sh need three arguments."
    echo "Usage: run.sh <path to context.json> <path to test.json> <path to output file>"
    exit 1
fi
echo "path to context.json: ${1}"
echo "path to test.json: ${2}"
echo "path to output file: ${3}"

context_file=$1
test_file=$2
prediction_file=$3

echo "Running Multiple choice."
python ./Code/Eval_MC.py --test_file "$test_file" --context_file "$context_file" --tokenizer_name hfl/chinese-bert-wwm --max_seq_length 512 --model_name_or_path ./Output_MC_final/ --output_dir ./
echo "Running Question Answering."
python ./Code/Eval_QA.py --test_file ./test_QA_Uni.json --tokenizer_name hfl/chinese-roberta-wwm-ext --model_name_or_path ./Output_QA_final/ --max_seq_length 512 --output_name "$prediction_file"