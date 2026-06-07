#!/usr/bin/env python3
"""Annotate image-expression prompts with GPT-4o."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_PROMPT_PATH = ROOT_DIR / "data/add_dataset/image_synthesis/annot_data/gpt_prompt.jsonl"
DEFAULT_SAVE_PATH = ROOT_DIR / "data/add_dataset/image_synthesis/annot_data/gpt_result.jsonl"
DEFAULT_API_CONFIG = Path("/home/jihyunlee/mirror/MIRROR_code/configs/api.json")


def load_api_key(api_config: Path | None) -> str | None:
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    if api_config and api_config.exists():
        with api_config.open(encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg.get("api-key") or cfg.get("api-key-personal")
    return None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--save-path", type=Path, default=DEFAULT_SAVE_PATH)
    parser.add_argument("--api-config", type=Path, default=DEFAULT_API_CONFIG)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--temperature", type=float, default=0.4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = load_api_key(args.api_config)
    client = OpenAI(api_key=api_key)
    prompts = load_jsonl(args.prompt_path)
    completed_ids = {entry["custom_id"] for entry in load_jsonl(args.save_path)}
    args.save_path.parent.mkdir(parents=True, exist_ok=True)

    with args.save_path.open("a", encoding="utf-8") as f:
        for prompt in tqdm(prompts, desc="Annotate expressions with GPT"):
            if prompt["custom_id"] in completed_ids:
                continue

            response = client.chat.completions.create(
                model=args.model,
                messages=prompt["messages"],
                temperature=args.temperature,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            payload = json.loads(content)
            entry = {
                "custom_id": prompt["custom_id"],
                "row_idx": prompt.get("row_idx", ""),
                "turn_index": prompt.get("turn_index", ""),
                "model": args.model,
                "messages": prompt["messages"],
                "response": {
                    "facial_expression_description": payload.get("facial_expression_description", ""),
                    "contrasting_facial_expression_description": payload.get("contrasting_facial_expression_description", ""),
                },
            }
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")
            f.flush()

    print(f"Wrote annotations: {args.save_path}")


if __name__ == "__main__":
    main()
