#!/usr/bin/env python3
"""Convert GPT expression annotations into PhotoMaker prompt JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_DATA_PATH = ROOT_DIR / "data/raw/add_dataset.csv"
DEFAULT_GPT_RESULT_PATH = ROOT_DIR / "data/add_dataset/image_synthesis/annot_data/gpt_result.jsonl"
DEFAULT_SAVE_PATH = ROOT_DIR / "data/add_dataset/image_synthesis/photomaker_prompts/prompt.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def gender_for_row(row: pd.Series) -> str:
    value = str(row.get("dominant_gender", "")).strip().lower()
    if value in {"man", "male"}:
        return "male"
    if value in {"woman", "female"}:
        return "female"
    gender_blob = str(row.get("gender", "")).lower()
    if "man" in gender_blob and "woman" not in gender_blob:
        return "male"
    return "female"


def custom_id_parts(custom_id: str) -> tuple[str, str]:
    if "-turn:" not in custom_id:
        raise ValueError(f"Unexpected custom_id: {custom_id}")
    row_idx, turn = custom_id.rsplit("-turn:", 1)
    return row_idx, f"turn:{turn}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--gpt-result-path", type=Path, default=DEFAULT_GPT_RESULT_PATH)
    parser.add_argument("--save-path", type=Path, default=DEFAULT_SAVE_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.data_path, dtype=str).fillna("")
    rows_by_idx = {row["idx"]: row for _, row in df.iterrows()}
    results = load_jsonl(args.gpt_result_path)
    args.save_path.parent.mkdir(parents=True, exist_ok=True)

    with args.save_path.open("w", encoding="utf-8") as f:
        for entry in tqdm(results, desc="Build PhotoMaker prompts"):
            row_idx, _ = custom_id_parts(entry["custom_id"])
            if row_idx not in rows_by_idx:
                continue

            row = rows_by_idx[row_idx]
            response = entry.get("response", {})
            expression = response.get("facial_expression_description", "").strip()
            contrast = response.get("contrasting_facial_expression_description", "").strip()
            gender = gender_for_row(row)
            image_path = str(row["img_path"])

            prompt = (
                f"portrait photo of a {gender} img, perfect face, natural skin, high detail, "
                f"counseling session, realistic lighting, {expression}"
            )
            negative_prompt = (
                "nsfw, lowres, bad anatomy, bad hands, grayscale photograph, text, error, "
                "missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, "
                "normal quality, jpeg artifacts, signature, watermark, username, blurry, "
                f"{contrast}"
            )
            output = {
                "idx": entry["custom_id"],
                "dialog_idx": row_idx,
                "image_path": [image_path],
                "prompt": prompt,
                "negative_prompt": negative_prompt,
            }
            json.dump(output, f, ensure_ascii=False)
            f.write("\n")

    print(f"Wrote PhotoMaker prompts: {args.save_path}")


if __name__ == "__main__":
    main()
