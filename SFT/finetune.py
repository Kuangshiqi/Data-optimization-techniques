import copy
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, Literal

import torch
import transformers
from transformers import Trainer
from datasets import load_dataset


def build_instruction_prompt(instruction: str) -> str:
    return (
        "\nYou are an AI programming assistant, please generate Python code according to the following instruction.\n"
        "### Instruction:\n{}\n### Response:\n"
    ).format((instruction or "").strip()).lstrip()


HF_MODEL_REGISTRY: Dict[str, str] = {
    "qwen1.5": "Qwen/Qwen2.5-Coder-1.5B-Instruct",   
    "qwen3": "Qwen/Qwen2.5-Coder-3B-Instruct",         
    "llama1": "meta-llama/Llama-3.2-1B-Instruct",      
    "llama3": "meta-llama/Llama-3.2-3B-Instruct",   
}


@dataclass
class ModelArguments:
    model: Literal["qwen1.5", "qwen3", "llama1", "llama3"] = field(
        metadata={"help": "Which model family to finetune (one of 4)."}
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional override for HF model id/path. If not set, use built-in registry."},
    )


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data (jsonl)."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=1024,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    truncate_source: bool = field(default=False)


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]

    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(sources: Sequence[str], targets: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]

    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = -100
    return dict(input_ids=input_ids, labels=labels)


@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def _resolve_model_name_or_path(model_key: str, override: Optional[str]) -> str:
    if override:
        return override
    if model_key not in HF_MODEL_REGISTRY:
        raise ValueError(f"Unknown model={model_key}. expected one of {list(HF_MODEL_REGISTRY.keys())}")
    return HF_MODEL_REGISTRY[model_key]


def _load_tokenizer(model_key: str, model_name_or_path: str, training_args: TrainingArguments):
    if model_key in ("qwen1.5", "qwen3"):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            truncation=True,
            padding_side="right",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = "<|extra_0|>"
        if tokenizer.eos_token is None:
            tokenizer.eos_token = "<|im_end|>"
        tokenizer.add_special_tokens({"additional_special_tokens": ["<|im_end|>", "<|im_start|>"]})
        end_token = "</s>"
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            truncation=True,
            padding_side="right",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = "<|extra_0|>"
        tokenizer.add_special_tokens({"additional_special_tokens": ["<|eot_id|>", "<|begin_of_text|>"]})
        end_token = "<|eot_id|>"

    return tokenizer, end_token


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model_name_or_path = _resolve_model_name_or_path(model_args.model, model_args.model_name_or_path)
    tokenizer, end_token = _load_tokenizer(model_args.model, model_name_or_path, training_args)

    def train_tokenize_function(examples, tokenizer):
        sources = [build_instruction_prompt(instruction) for instruction in examples["question"]]
        targets = [f"{output}\n{end_token}" for output in examples["solutions"]]
        return preprocess(sources, targets, tokenizer)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
    )

    raw_train_datasets = load_dataset(
        "json",
        data_files={"train": f"{data_args.data_path}"},
        split="train",
        cache_dir=training_args.cache_dir,
    )

    train_dataset = raw_train_datasets.map(
        train_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=32,
        remove_columns=raw_train_datasets.column_names,
        load_from_cache_file=True,
        desc="Running Encoding",
        fn_kwargs={"tokenizer": tokenizer},
    )

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_state()


if __name__ == "__main__":
    train()
