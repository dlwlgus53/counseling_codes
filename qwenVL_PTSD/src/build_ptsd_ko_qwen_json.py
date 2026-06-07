#!/usr/bin/env python3
"""Build text-only Qwen SFT data from PTSD JSONL dialogues."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """You are a trauma-informed CBT counseling assistant.
Always respond in Korean.

Begin each session with a gentle, open-ended invitation to discuss the client’s current daily difficulty, rather than directly naming the trauma or target problem.
Help the client gradually disclose the avoided situation, the fear or interpretation behind it, and the behavior they use to avoid or escape it.
Keep the conversation natural, empathic, and collaborative; do not sound like a checklist or interrogation.
Validate the client’s distress as an understandable response to what happened, while carefully avoiding blame or minimization.
Use Socratic questioning, guided discovery, evidence-for/evidence-against questions, decatastrophizing, and past-present differentiation to support cognitive change.
Guide the client toward an adaptive thought that recognizes the reality of the past danger while also making room for present safety, self-efficacy, and gradual recovery.
When the client is ready, suggest one small, safe, and realistic behavioral practice task that reduces avoidance and supports adaptive behavior.
Avoid diagnosis, medication advice, forced positivity, or sudden/intense exposure.
Close the conversation by summarizing the client’s insight and confirming one manageable next step."""


def text_content(text: str) -> list[dict[str, str]]:
    return [{"type": "text", "text": text}]


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    warnings: list[str] = []

    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                warnings.append(f"{path.name}:{line_number}: invalid JSON: {exc}")
                continue
            if not isinstance(record, dict):
                warnings.append(f"{path.name}:{line_number}: top-level value is not an object")
                continue
            records.append(record)

    return records, warnings


def read_dataset(input_path: Path) -> tuple[list[dict[str, Any]], list[str], list[Path]]:
    if input_path.is_file():
        paths = [input_path]
    elif input_path.is_dir():
        paths = sorted(input_path.glob("*.jsonl"))
        if not paths:
            raise FileNotFoundError(f"No .jsonl files found in raw directory: {input_path}")
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    for path in paths:
        path_records, path_warnings = read_jsonl(path)
        records.extend(path_records)
        warnings.extend(path_warnings)
    return records, warnings, paths


def turn_role(turn: dict[str, Any]) -> str:
    return str(turn.get("role", "")).strip()


def turn_text(turn: dict[str, Any]) -> str:
    text = turn.get("utterance", "")
    if text is None:
        return ""
    return str(text).strip()


def history_text(turns: list[dict[str, Any]], target_index: int, context_len: int) -> str:
    lines: list[str] = []
    role_labels = {
        "counselor": "상담자",
        "client": "내담자",
    }
    for turn in turns[:target_index]:
        if not isinstance(turn, dict):
            continue
        role = turn_role(turn)
        text = turn_text(turn)
        if not role or not text:
            continue
        lines.append(f"{role_labels.get(role, role)}: {text}")

    if context_len > 0:
        lines = lines[-context_len:]
    return "\n".join(lines)


def prompt_for(history: str, current_turn: int) -> str:
    if history:
        history_block = history
    else:
        history_block = "이전 대화 없음"

    return "\n".join(
        [
            "아래 대화 히스토리만 바탕으로 상담자의 다음 응답을 작성하세요.",
            "",
            "[현재 상담 턴]",
            f"{current_turn}턴",
            "",
            "[대화 히스토리]",
            history_block,
        ]
    )


def build_examples(input_path: Path, context_len: int) -> tuple[list[dict[str, Any]], list[str], list[Path]]:
    examples: list[dict[str, Any]] = []
    records, warnings, input_files = read_dataset(input_path)

    for record_index, data in enumerate(records, start=1):
        sample_id = str(data.get("news_id") or data.get("client_id") or f"record_{record_index}")
        turns = data.get("dialogue")
        if not isinstance(turns, list):
            warnings.append(f"{sample_id}: dialogue is not a list")
            continue

        counselor_turn = 0
        for turn_index, turn in enumerate(turns):
            if not isinstance(turn, dict):
                warnings.append(f"{sample_id}: utterance {turn_index + 1} is not an object")
                continue

            role = turn_role(turn)
            if role == "counselor":
                counselor_turn += 1
            if role != "counselor":
                continue

            counselor = turn_text(turn)
            if not counselor:
                warnings.append(f"{sample_id}: counselor turn {counselor_turn} has empty utterance")
                continue

            current_turn = counselor_turn
            history = history_text(turns, target_index=turn_index, context_len=context_len)
            examples.append(
                {
                    "id": f"{sample_id}:{current_turn}:counselor",
                    "source_id": sample_id,
                    "turn": current_turn,
                    "utterance_index": turn_index + 1,
                    "messages": [
                        {"role": "system", "content": text_content(SYSTEM_PROMPT)},
                        {"role": "user", "content": text_content(prompt_for(history, current_turn))},
                        {"role": "assistant", "content": text_content(counselor)},
                    ],
                }
            )

    return examples, warnings, input_files


def write_jsonl(output_path: Path, examples: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False))
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/home/jihyunlee/mirror/MIRROR_code/qwenVL_PTSD/data/raw"),
        help="A PTSD JSONL file or a directory containing raw *.jsonl files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/jihyunlee/mirror/MIRROR_code/qwenVL_PTSD/data/ptsd_ko_train.jsonl"),
    )
    parser.add_argument(
        "--context-len",
        type=int,
        default=10,
        help="Maximum previous utterances to include. Counselor and client lines each count as one utterance.",
    )
    args = parser.parse_args()

    examples, warnings, input_files = build_examples(input_path=args.input, context_len=args.context_len)
    write_jsonl(args.output, examples)

    print(f"Input files: {len(input_files)}")
    for input_file in input_files:
        print(f"  - {input_file}")
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
