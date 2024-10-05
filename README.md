# 2024ADL Homework1

## 環境與套件
程式語言：Python 3.10.12<br>
套件：PyTorch 2.1.0 scikit-learn 1.5.1 transformers 4.44.2, datasets 2.21.0, accelerate 0.34.2 evaluate 0.4.0 tqdm 4.66.5 numpy 1.26.4 pandas 2.2.2  matplotlib 3.9.2<br>
作業系統：Windows 11 64bit
## 使用方式
### Step1. Multiple choice:
1. 訓練使用Code/run_swag_no_trainer.py<br>
參數(以下皆以表現最好的模型為例子)：<br>
```
python ./Code/run_swag_no_trainer.py --train_file ./Dataset/train.json --validation_file ./Dataset/valid.json --context_file ./Dataset/context.json --output_dir ./Output_MC --num_train_epoch 1 --max_seq_length 512 --model_name_or_path hfl/chinese-bert-wwm --tokenizer_name hfl/chinese-bert-wwm --per_device_train_batch_size 1 --per_device_eval_batch_size 1  --gradient_accumulation_steps 2 --learning_rate 3e-5
```
2. 預測使用Code/Eval_MC.py<br>
參數：<br>
```
python ./Code/Eval_MC.py --test_file ./Dataset/test.json --context_file ./Dataset/context.json --tokenizer_name ./Output_MC --max_seq_length 512 --model_name_or_path ./Output_MC --output_dir ./Dataset
```
### Step2. Question answering:
1. 訓練使用Code/run_qa_no_trainer.py<br>
參數：<br>
```
python ./Code/run_qa_no_trainer.py --train_file ./Dataset/train.json --validation_file ./Dataset/valid.json --context_file ./Dataset/context.json --output_dir ./Output_QA --num_train_epoch 2 --max_seq_length 512 --model_name_or_path hfl/chinese-roberta-wwm-ext --tokenizer_name hfl/chinese-roberta-wwm-ext --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --gradient_accumulation_steps 2 --learning_rate 3e-5 --lr_scheduler_type cosine
```
2. 預測使用Code/Eval_QA.py<br>
參數：<br>
```
python ./Code/Eval_QA.py --test_file ./Dataset/test_QA_Unicode.json --tokenizer_name ./Output_QA --model_name_or_path ./Output_QA --max_seq_length 512 --output_name ./prediction.csv
```
## Draw plot:
使用drawplot.ipynb
## download.sh & run.sh:
讓助教測試用的，download.sh會下載模型，run.sh會跑預測<br>
其中run.sh需要三個參數:<br>
```
"${1}": path to context.json.
"${2}": path to test.json.
"${3}": path to the output prediction file named prediction.csv.
```
## Reference:
此作業是基於以下GitHub repo改寫：<br>
https://github.com/huggingface/transformers/tree/main