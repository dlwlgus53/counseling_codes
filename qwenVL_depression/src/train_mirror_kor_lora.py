#!/usr/bin/env python3
"""LoRA SFT for Qwen3-VL on MIRROR Korean multimodal chat data."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


IGNORE_INDEX = -100


def normalize_content(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return content
    return [{"type": "text", "text": str(content)}]


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **message,
            "content": normalize_content(message.get("content", "")),
        }
        for message in messages
    ]


class MirrorQwenDataset(Dataset):
    def __init__(self, data_path: str | Path) -> None:
        self.examples: list[dict[str, Any]] = []
        with Path(data_path).open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]


@dataclass
class QwenVlCollator:
    processor: Any
    max_length: int

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        features = [self._encode(example) for example in examples]
        return self._pad_and_merge(features)

    def _encode(self, example: dict[str, Any]) -> dict[str, torch.Tensor]:
        messages = normalize_messages(example["messages"])
        prompt_messages = messages[:-1]

        full = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )
        prompt = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        input_ids = full["input_ids"][0]
        if input_ids.numel() > self.max_length:
            input_ids = input_ids[: self.max_length]

        labels = input_ids.clone()
        prompt_len = min(prompt["input_ids"].shape[-1], labels.shape[-1])
        labels[:prompt_len] = IGNORE_INDEX

        feature: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": labels,
        }

        for key, value in full.items():
            if key in feature or key == "input_ids" or value is None:
                continue
            if torch.is_tensor(value):
                tensor = value
                if tensor.ndim >= 1 and tensor.shape[0] == 1 and key not in {"pixel_values", "image_grid_thw"}:
                    tensor = tensor[0]
                feature[key] = tensor

        return feature

    def _pad_and_merge(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.processor.tokenizer.eos_token_id

        max_len = max(feature["input_ids"].shape[0] for feature in features)
        batch: dict[str, torch.Tensor] = {}

        for key, pad_value in (
            ("input_ids", pad_id),
            ("attention_mask", 0),
            ("labels", IGNORE_INDEX),
        ):
            padded = []
            for feature in features:
                value = feature[key]
                pad_len = max_len - value.shape[0]
                if pad_len:
                    value = torch.nn.functional.pad(value, (0, pad_len), value=pad_value)
                padded.append(value)
            batch[key] = torch.stack(padded, dim=0)

        extra_keys = sorted(set().union(*(feature.keys() for feature in features)) - set(batch.keys()))
        for key in extra_keys:
            values = [feature[key] for feature in features if key in feature]
            if not values or not all(torch.is_tensor(value) for value in values):
                continue
            if key in {"pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"}:
                batch[key] = torch.cat(values, dim=0)
            else:
                try:
                    batch[key] = torch.stack(values, dim=0)
                except RuntimeError:
                    batch[key] = torch.cat(values, dim=0)

        return batch


class IOSampleLoggerCallback(TrainerCallback):
    def __init__(
        self,
        processor: Any,
        dataset: MirrorQwenDataset,
        output_dir: str | Path,
        every_steps: int,
        num_samples: int,
        max_new_tokens: int,
        generate_outputs: bool,
    ) -> None:
        self.processor = processor
        self.dataset = dataset
        self.output_dir = Path(output_dir)
        self.every_steps = every_steps
        self.num_samples = num_samples
        self.max_new_tokens = max_new_tokens
        self.generate_outputs = generate_outputs
        self.logged_steps: set[int] = set()

    def on_log(self, args: TrainingArguments, state: Any, control: Any, **kwargs: Any) -> None:
        step = int(state.global_step)
        if step <= 0 or step in self.logged_steps:
            return
        if self.every_steps <= 0 or step % self.every_steps != 0:
            return
        if not state.is_world_process_zero:
            return

        self.logged_steps.add(step)
        model = kwargs.get("model")
        if model is None:
            return

        records = []
        for sample_offset in range(self.num_samples):
            index = (step - 1 + sample_offset) % len(self.dataset)
            example = self.dataset[index]
            record = self._build_record(model=model, example=example, step=step, index=index)
            records.append(record)
            self._print_record(record)

        self._write_records(records)
        self._log_to_wandb(records, step)

    def _build_record(self, model: Any, example: dict[str, Any], step: int, index: int) -> dict[str, Any]:
        messages = normalize_messages(example["messages"])
        prompt_messages = messages[:-1]
        prompt_text = self._content_text(prompt_messages[0]["content"])
        gold_text = self._content_text(messages[-1]["content"])
        generated_text = ""
        generation_error = ""

        if self.generate_outputs:
            try:
                generated_text = self._generate(model=model, prompt_messages=prompt_messages)
            except Exception as exc:  # noqa: BLE001 - logging should not stop training.
                generation_error = repr(exc)

        return {
            "step": step,
            "dataset_index": index,
            "id": example.get("id"),
            "image": example.get("image"),
            "prompt": prompt_text,
            "gold_answer": gold_text,
            "generated_answer": generated_text,
            "generation_error": generation_error,
        }

    def _generate(self, model: Any, prompt_messages: list[dict[str, Any]]) -> str:
        unwrapped_model = model.module if hasattr(model, "module") else model
        was_training = unwrapped_model.training
        unwrapped_model.eval()
        inputs = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        device = next(unwrapped_model.parameters()).device
        inputs = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }
        prompt_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            output_ids = unwrapped_model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        generated_ids = output_ids[0, prompt_len:]
        generated_text = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        if was_training:
            unwrapped_model.train()
        return generated_text

    def _write_records(self, records: list[dict[str, Any]]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with (self.output_dir / "io_samples.jsonl").open("a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False))
                f.write("\n")

    def _log_to_wandb(self, records: list[dict[str, Any]], step: int) -> None:
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is None:
            return
        rows = [
            [
                record["step"],
                record["dataset_index"],
                record["id"],
                record["image"],
                record["prompt"],
                record["gold_answer"],
                record["generated_answer"],
                record["generation_error"],
            ]
            for record in records
        ]
        table = wandb.Table(
            columns=[
                "step",
                "dataset_index",
                "id",
                "image",
                "prompt",
                "gold_answer",
                "generated_answer",
                "generation_error",
            ],
            data=rows,
        )
        wandb.log({"io_samples": table}, step=step)

    @staticmethod
    def _content_text(content: str | list[dict[str, Any]]) -> str:
        if isinstance(content, str):
            return content
        parts = []
        for item in content:
            if item.get("type") == "image":
                parts.append(f"[image] {item.get('image', '')}")
            elif item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(part for part in parts if part)

    @staticmethod
    def _print_record(record: dict[str, Any]) -> None:
        print("\n==== IO SAMPLE ====", flush=True)
        print(f"step: {record['step']} | dataset_index: {record['dataset_index']} | id: {record['id']}", flush=True)
        print(f"image: {record['image']}", flush=True)
        print("---- prompt ----", flush=True)
        print(record["prompt"], flush=True)
        print("---- gold answer ----", flush=True)
        print(record["gold_answer"], flush=True)
        print("---- generated answer ----", flush=True)
        print(record["generated_answer"] or f"[generation skipped/error] {record['generation_error']}", flush=True)
        print("===================\n", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--num_train_epochs", type=float, default=3)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--io_log_steps", type=int, default=50)
    parser.add_argument("--io_log_num_samples", type=int, default=1)
    parser.add_argument("--io_log_max_new_tokens", type=int, default=128)
    parser.add_argument("--io_log_no_generate", action="store_true")
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--use_flash_attention_2", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    parser.add_argument("--report_to", default="none")
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    return parser.parse_args()


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def disable_deepspeed_auto_import() -> None:
    try:
        import accelerate.utils.other as accelerate_other
    except ImportError:
        return

    # This trainer path does not use DeepSpeed, but Accelerate still imports it
    # when the package is installed. On this env that import requires CUDA_HOME.
    accelerate_other.is_deepspeed_available = lambda: False


def save_run_arguments(args: argparse.Namespace, output_dir: str | Path, world_size: int) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    payload = vars(args).copy()
    payload["world_size"] = world_size
    payload["effective_train_batch_size"] = (
        args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size
    )
    payload["environment"] = {
        "WORLD_SIZE": os.environ.get("WORLD_SIZE"),
        "RANK": os.environ.get("RANK"),
        "LOCAL_RANK": os.environ.get("LOCAL_RANK"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "WANDB_PROJECT": os.environ.get("WANDB_PROJECT"),
        "WANDB_RUN_NAME": os.environ.get("WANDB_RUN_NAME"),
        "WANDB_MODE": os.environ.get("WANDB_MODE"),
    }
    with (output_path / "training_args.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def print_batch_summary(args: argparse.Namespace, dataset_size: int, world_size: int) -> None:
    effective_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size
    if not is_main_process():
        return
    print("==== MIRROR Qwen3-VL LoRA Training ====", flush=True)
    print(f"Dataset examples: {dataset_size}", flush=True)
    print(f"GPUs / world size: {world_size}", flush=True)
    print(f"Per-device train batch size: {args.per_device_train_batch_size}", flush=True)
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}", flush=True)
    print(f"Effective train batch size: {effective_batch}", flush=True)
    print(f"Output dir: {args.output_dir}", flush=True)
    print(f"Report to: {args.report_to}", flush=True)
    print("========================================", flush=True)


def main() -> None:
    args = parse_args()
    disable_deepspeed_auto_import()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else "auto"),
        "quantization_config": quantization_config,
    }
    if args.load_in_4bit:
        model_kwargs["device_map"] = {"": local_rank}
    if args.use_flash_attention_2:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForImageTextToText.from_pretrained(args.model_name_or_path, **model_kwargs)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.lora_target_modules,
    )
    model = get_peft_model(model, lora_config)
    if is_main_process():
        model.print_trainable_parameters()

    train_dataset = MirrorQwenDataset(args.data_path)
    collator = QwenVlCollator(processor=processor, max_length=args.max_length)
    print_batch_summary(args=args, dataset_size=len(train_dataset), world_size=world_size)
    if is_main_process():
        save_run_arguments(args=args, output_dir=args.output_dir, world_size=world_size)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        run_name=args.run_name,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,
        report_to=args.report_to,
        dataloader_num_workers=args.dataloader_num_workers,
        ddp_find_unused_parameters=False,
        optim="paged_adamw_8bit" if args.load_in_4bit else "adamw_torch",
        lr_scheduler_type="cosine",
    )

    callbacks = []
    if args.io_log_steps > 0 and args.io_log_num_samples > 0:
        callbacks.append(
            IOSampleLoggerCallback(
                processor=processor,
                dataset=train_dataset,
                output_dir=args.output_dir,
                every_steps=args.io_log_steps,
                num_samples=args.io_log_num_samples,
                max_new_tokens=args.io_log_max_new_tokens,
                generate_outputs=not args.io_log_no_generate,
            )
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
