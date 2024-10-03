import argparse
import json
from dataclasses import dataclass
from itertools import chain
from typing import Optional, Union

import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader

from transformers import (
    AutoModelForMultipleChoice,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)
from transformers.utils import PaddingStrategy

def parse_args():
    parser = argparse.ArgumentParser(description="Use a transformers model on a multiple choice task")
    parser.add_argument(
        "--test_file", type=str, required=True, help="A csv or a json file containing the test data."
    )
    parser.add_argument(
        "--context_file", type=str, required=True, help="A csv or a json file containing the context data."
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument("--output_dir", type=str, default="./", help="Where to store the final prediction.")
    args = parser.parse_args()

    return args

@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              if provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = list(chain(*flattened_features))

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        return batch

def main():
    args = parse_args()
    # read model
    accelerator = Accelerator(gradient_accumulation_steps=2)
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=not args.use_slow_tokenizer
        )
    model = AutoModelForMultipleChoice.from_pretrained(args.model_name_or_path)

    device = accelerator.device
    model.to(device)
    
    extension = args.test_file.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=args.test_file)
    if args.context_file is not None:
        with open(args.context_file, 'r', encoding="utf-8") as data_context:
            data_context = json.load(data_context)
    else:
        data_context = []
    
    def addends(examples):
        i = 0
        for pid in examples["paragraphs"]:
            examples["ending{}".format(i)] = data_context[pid]
            i += 1
        return examples
    modify_datasets = raw_datasets.map(addends)
    ending_names = [f"ending{i}" for i in range(4)]
    context_name = "question" #sent1
    question_header_name = "question" #sent2

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
        
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    padding = True

    def preprocess_function(examples):
        first_sentences = [[context] * 4 for context in examples[context_name]]
        question_headers = examples[question_header_name]
        second_sentences = [
            [f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
        ]
        # Flatten out
        first_sentences = list(chain(*first_sentences))
        second_sentences = list(chain(*second_sentences))

        # Tokenize
        tokenized_examples = tokenizer(
            first_sentences,
            second_sentences,
            max_length=args.max_seq_length,
            padding=padding,
            truncation=True,
            return_tensors="pt"
        )
        # Un-flatten
        tokenized_inputs = {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
        # tokenized_inputs = tokenized_examples
        return tokenized_inputs
    
    processed_datasets = modify_datasets.map(
        preprocess_function, batched=True, remove_columns=modify_datasets["train"].column_names)
    data_collator = DataCollatorForMultipleChoice(tokenizer)

    eval_dataloader = DataLoader(processed_datasets['train'], collate_fn=data_collator, batch_size=1)
    eval_dataloader = accelerator.prepare(eval_dataloader)
    predictions = []
    model.eval()
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
        predictions.append(outputs.logits.argmax(dim=-1))
    def postprocess_function(examples, idx):
        examples["context"] = data_context[examples["paragraphs"][predictions[idx]]]
        return examples
    predictions_final = raw_datasets.map(postprocess_function, with_indices=True) 
    # 使用 json.dump 寫入文件，並設定 ensure_ascii=False by ChatGPT
    data_list = [item for item in predictions_final["train"]]
    with open(f'{args.output_dir}test_QA_Unicode.json', 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()