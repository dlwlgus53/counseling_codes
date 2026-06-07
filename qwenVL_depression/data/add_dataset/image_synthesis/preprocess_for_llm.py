#!/usr/bin/env python3
"""Build GPT prompt JSONL for add_dataset image-expression generation."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_DATA_PATH = ROOT_DIR / "data/raw/add_dataset.csv"
DEFAULT_SAVE_PATH = ROOT_DIR / "data/add_dataset/image_synthesis/annot_data/gpt_prompt.jsonl"

SYSTEM_MESSAGE = (
    "You are an AI assistant for image generation. You are given a counseling "
    "conversation history and the client's current utterance. Describe the "
    "client's facial expression in English for image generation. Also provide "
    "a contrasting facial expression. Return only JSON."
)


def parse_proc_dialogue(value: Any) -> list[dict[str, str]]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
    raise ValueError(f"Unsupported proc_dialogue value: {type(value)}")


def speaker_name(raw: str) -> str:
    return "Therapist" if raw.strip().lower() == "therapist" else "Client"


def turn_to_text(turn: dict[str, str]) -> str:
    speaker = speaker_name(turn.get("speaker", "Client"))
    stage_direction = str(turn.get("stage_direction", "")).strip()
    statement = str(turn.get("statement", "")).strip()
    if stage_direction:
        return f"{speaker}: [{stage_direction}] {statement}"
    return f"{speaker}: {statement}"


def build_dialogue_history(proc_dialogue: list[dict[str, str]]) -> str:
    return "\n".join(turn_to_text(turn) for turn in proc_dialogue).strip()


def build_messages(history: str, client_utterance: str) -> list[dict[str, str]]:
    user = f"""
Create English image-generation descriptions for the client's current expression.

Requirements:
- Focus on visible facial expression, gaze, posture, and emotional tone.
- Do not describe the therapist.
- Do not include diagnosis or mental-health labels.
- Keep both descriptions concise and visual.
- Return JSON with exactly these keys:
  facial_expression_description
  contrasting_facial_expression_description

Dialogue History:
{history}

Client's Current Utterance:
{client_utterance}
"""
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user},
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--save-path", type=Path, default=DEFAULT_SAVE_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.data_path, dtype=str).fillna("")
    args.save_path.parent.mkdir(parents=True, exist_ok=True)

    with args.save_path.open("w", encoding="utf-8") as f:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Build GPT image prompts"):
            dialog = parse_proc_dialogue(row["proc_dialogue"])
            for turn_index, turn in enumerate(dialog):
                if speaker_name(turn.get("speaker", "Client")) != "Client":
                    continue

                history = build_dialogue_history(dialog[:turn_index])
                client_utterance = turn_to_text(turn)
                entry = {
                    "custom_id": f"{row['idx']}-turn:{turn_index}",
                    "row_idx": row["idx"],
                    "turn_index": turn_index,
                    "messages": build_messages(history=history, client_utterance=client_utterance),
                }
                json.dump(entry, f, ensure_ascii=False)
                f.write("\n")

    print(f"Wrote prompts: {args.save_path}")


if __name__ == "__main__":
    main()
