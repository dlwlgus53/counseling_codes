#!/usr/bin/env python3
"""Build Qwen3-VL MIRROR Korean SFT data.

The output is JSONL in the Hugging Face/Qwen chat-template style:

{"messages": [
  {"role": "user", "content": [
    {"type": "image", "image": "/abs/path.png"},
    {"type": "text", "text": "..."}
  ]},
  {"role": "assistant", "content": [
    {"type": "text", "text": "..."}
  ]}
]}

Qwen's official examples use multimodal content blocks instead of putting the
image placeholder directly in the user text when using `apply_chat_template`.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

IMAGE_DESC = "위 이미지는 내담자를 보여줍니다."
CBT_DESC = (
    "내담자의 몸짓과 표정을 바탕으로, CBT(인지행동치료) 세션을 진행하는 심리치료사처럼 응답하세요. "
    "응답은 반드시 먼저 [클라이언트 행동: 직전 내담자의 행동, 표정, 목소리 톤] 형식으로 시작한 뒤 "
    "치료사의 발화를 이어서 작성하세요. 대괄호 안에는 치료사의 행동이 아니라 클라이언트의 모습만 적으세요."
)

SPEAKER_MAP = {
    "Therapist": "치료사",
    "Client": "내담자",
    "치료사": "치료사",
    "상담사": "치료사",
    "심리치료사": "치료사",
    "클라이언트": "내담자",
    "내담자": "내담자",
}


def parse_dialogue(dialogue: str) -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    for raw_line in dialogue.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue

        raw_speaker, rest = line.split(":", 1)
        speaker = SPEAKER_MAP.get(raw_speaker.strip(), "내담자")

        rest = rest.strip()
        stage_direction = ""
        if rest.startswith("[") and "]" in rest:
            stage_direction, rest = rest[1:].split("]", 1)
            rest = rest.strip()

        turns.append(
            {
                "speaker": speaker,
                "stage_direction": stage_direction.strip(),
                "statement": rest,
            }
        )
    return turns


def history_text(turns: list[dict[str, str]], context_len: int) -> str:
    if context_len > 0:
        turns = turns[-context_len:]
    return "\n".join(f"{turn['speaker']}: {turn['statement']}" for turn in turns)


def previous_client_stage_direction(turns: list[dict[str, str]], turn_index: int) -> str:
    for previous_turn in reversed(turns[:turn_index]):
        if previous_turn["speaker"] == "내담자":
            return previous_turn.get("stage_direction", "").strip()
    return "첫 번째 턴이라 이전 클라이언트 행동 없음"


def response_text(turns: list[dict[str, str]], turn_index: int) -> str:
    stage_direction = previous_client_stage_direction(turns, turn_index)
    statement = turns[turn_index]["statement"].strip()
    if stage_direction:
        return f"[클라이언트 행동: {stage_direction}] {statement}"
    return statement


def prompt_for(history: str, current_turn: int) -> str:
    parts = [
        IMAGE_DESC,
        f"- 현재 상담 턴: {current_turn}턴",
    ]
    if history:
        parts.extend(
            [
                "",
                "아래는 내담자와 심리치료사의 이전 대화입니다. 최대 최근 10개 발화만 포함됩니다.",
                history,
                "",
            ]
        )
    else:
        parts.extend(
            [
                "",
                "이것은 첫 턴입니다. 이전 대화는 없습니다.",
                "",
            ]
        )
    parts.append(CBT_DESC)
    return "\n".join(parts)


def image_for_turn(row: dict[str, str], turn_index: int, image_root: Path) -> Path:
    source_image = Path(row["img_path"])
    image_id = source_image.stem
    image_turn = max(1, turn_index - 1)
    return image_root / row["idx"] / f"turn:{image_turn}_{image_id}.png"


def safe_proc_dialogue_len(row: dict[str, str]) -> int | None:
    raw = row.get("proc_dialogue", "")
    if not raw:
        return None
    try:
        parsed = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return None
    if isinstance(parsed, list):
        return len(parsed)
    return None


def build_examples(
    rows: list[dict[str, str]],
    image_root: Path,
    context_len: int,
    skip_missing_images: bool,
    skip_mismatched_turns: bool,
) -> tuple[list[dict[str, object]], list[str]]:
    examples: list[dict[str, object]] = []
    warnings: list[str] = []

    for row in rows:
        turns = parse_dialogue(row["dialogue"])
        proc_len = safe_proc_dialogue_len(row)
        if proc_len is not None and proc_len != len(turns):
            warnings.append(
                f"{row['idx']}: parsed {len(turns)} Korean turns, proc_dialogue has {proc_len}"
            )
            if skip_mismatched_turns:
                continue

        required_images = [
            image_for_turn(row, turn_index, image_root)
            for turn_index, turn in enumerate(turns)
            if turn["speaker"] == "치료사"
        ]
        missing_images = [image_path for image_path in required_images if not image_path.exists()]
        if skip_missing_images and missing_images:
            warnings.append(
                f"{row['idx']}: skipped row because {len(missing_images)} required images are missing"
            )
            continue

        for turn_index, turn in enumerate(turns):
            if turn["speaker"] != "치료사":
                continue

            image_path = image_for_turn(row, turn_index, image_root)
            history = history_text(turns[:turn_index], context_len=context_len)
            current_turn = sum(1 for past_turn in turns[: turn_index + 1] if past_turn["speaker"] == "치료사")
            prompt = prompt_for(history, current_turn)

            examples.append(
                {
                    "id": f"{row['idx']}:{turn_index}:response",
                    "image": str(image_path),
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": str(image_path)},
                                {"type": "text", "text": prompt},
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": response_text(turns, turn_index)},
                            ],
                        },
                    ],
                }
            )

    return examples, warnings


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_jsonl(output_path: Path, examples: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False))
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=ROOT_DIR / "data/raw/train_ko.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT_DIR / "data/mirror_ko_train.jsonl",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=ROOT_DIR / "data/mirror_images",
    )
    parser.add_argument(
        "--context-len",
        type=int,
        default=10,
        help="Maximum number of previous utterances to include. Therapist and client lines each count as one turn.",
    )
    parser.add_argument("--keep-missing-images", action="store_true")
    parser.add_argument("--keep-mismatched-turns", action="store_true")
    args = parser.parse_args()

    rows = read_rows(args.csv)
    examples, warnings = build_examples(
        rows=rows,
        image_root=args.image_root,
        context_len=args.context_len,
        skip_missing_images=not args.keep_missing_images,
        skip_mismatched_turns=not args.keep_mismatched_turns,
    )
    write_jsonl(args.output, examples)

    print(f"Read rows: {len(rows)}")
    print(f"Wrote examples: {len(examples)}")
    print(f"Output: {args.output}")
    if warnings:
        print(f"Warnings: {len(warnings)}")
        for warning in warnings[:20]:
            print(f"  - {warning}")
        if len(warnings) > 20:
            print(f"  ... {len(warnings) - 20} more")


if __name__ == "__main__":
    main()
