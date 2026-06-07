#!/usr/bin/env python3
"""Build text-only Qwen SFT data from panic raw JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "당신은 공황 발작, 급성 불안, 신체 증상으로 위기감을 느끼는 사용자를 돕는 "
    "한국어 응급상담 지원 모델입니다. 내담자의 고통을 먼저 공감하고 정서적으로 확인해 준 뒤, "
    "현재 위치와 안전을 파악하고, 사람이 많거나 붐비는 곳에 있다면 사람이 적은 모서리, 벽 근처, "
    "조용한 공간, 또는 가능한 경우 야외처럼 더 안전하고 덜 자극적인 곳으로 이동하도록 안내하세요. "
    "그다음 호흡 조절이나 그라운딩을 통해 몸을 안정시키세요. 내담자가 죽을 것 같다고 말하면 "
    "그 공포가 매우 실제처럼 느껴질 수 있음을 충분히 인정하면서, 많은 경우 심장이 빨리 뛰고 몸이 "
    "위협 반응을 보이는 것이지 곧 죽는다는 뜻은 아니라고 차분히 설명하세요. 단, 의학적 진단을 "
    "단정하지 말고 심한 호흡곤란, 흉통, 의식저하, 자해/타해 위험, 증상 악화가 있으면 119 또는 "
    "가까운 응급실 등 실제 응급 도움을 권유하세요. 내담자가 안정되었다고 말하면 세션을 짧게 "
    "요약하고, 오늘 너무 잘하셨다고 격려한 뒤, 증상이 지속되거나 반복되면 병원이나 전문가 상담을 "
    "받아보라고 안내하며 마무리하세요."
)


def text_content(text: str) -> list[dict[str, str]]:
    return [{"type": "text", "text": text}]


def history_text(turns: list[dict[str, Any]], counselor_turn_index: int, context_len: int) -> str:
    lines: list[str] = []
    for turn in turns[:counselor_turn_index]:
        counselor = str(turn.get("counselor", "")).strip()
        client = str(turn.get("client", "")).strip()
        if counselor:
            lines.append(f"상담사: {counselor}")
        if client:
            lines.append(f"내담자: {client}")

    if context_len > 0:
        lines = lines[-context_len:]
    return "\n".join(lines)


def prompt_for(
    *,
    history: str,
    current_turn: int,
) -> str:
    parts = [
        "아래의 현재까지 대화를 바탕으로 응급상담을 지원하는 상담사처럼 다음 응답을 작성하세요.",
    ]

    parts.extend(["", f"[현재 상담 턴]\n{current_turn}턴"])

    if history:
        parts.extend(
            [
                "",
                "[이전 대화]",
                history,
            ]
        )
    else:
        parts.extend(
            [
                "",
                "[이전 대화]",
                "첫 상담사 응답입니다. 아직 이전 대화는 없습니다.",
            ]
        )

    return "\n".join(parts)


def build_examples(
    dataset: dict[str, Any],
    context_len: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    examples: list[dict[str, Any]] = []
    warnings: list[str] = []

    for sample_id, sample in dataset.items():
        dialogue_block = sample.get("dialogue", {})
        turns = dialogue_block.get("dialogue", [])
        if not isinstance(turns, list):
            warnings.append(f"{sample_id}: dialogue.dialogue is not a list")
            continue

        for turn_index, turn in enumerate(turns):
            counselor = str(turn.get("counselor", "")).strip()
            if not counselor:
                warnings.append(f"{sample_id}: turn {turn_index + 1} has empty counselor text")
                continue

            current_turn = int(turn.get("turn", turn_index + 1))
            history = history_text(turns, counselor_turn_index=turn_index, context_len=context_len)
            prompt = prompt_for(
                history=history,
                current_turn=current_turn,
            )
            examples.append(
                {
                    "id": f"{sample_id}:{current_turn}:counselor",
                    "source_id": sample_id,
                    "turn": current_turn,
                    "messages": [
                        {"role": "system", "content": text_content(SYSTEM_PROMPT)},
                        {"role": "user", "content": text_content(prompt)},
                        {"role": "assistant", "content": text_content(counselor)},
                    ],
                }
            )

    return examples, warnings


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected top-level JSON object, got {type(data).__name__}")
    return data


def unique_sample_id(sample_id: str, dataset: dict[str, Any]) -> str:
    if sample_id not in dataset:
        return sample_id
    suffix = 2
    while f"{sample_id}-{suffix}" in dataset:
        suffix += 1
    return f"{sample_id}-{suffix}"


def read_dataset(input_path: Path) -> tuple[dict[str, Any], list[Path]]:
    if input_path.is_file():
        return read_json(input_path), [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    json_paths = sorted(input_path.glob("*.json"))
    if not json_paths:
        raise FileNotFoundError(f"No .json files found in raw directory: {input_path}")

    dataset: dict[str, Any] = {}
    for json_path in json_paths:
        for sample_id, sample in read_json(json_path).items():
            dataset[unique_sample_id(sample_id, dataset)] = sample
    return dataset, json_paths


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
        default=Path("/home/jihyunlee/mirror/MIRROR_code/qwenVL_panic/data/raw"),
        help="A panic dataset JSON file or a directory containing raw *.json files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/jihyunlee/mirror/MIRROR_code/qwenVL_panic/data/panic_ko_train.jsonl"),
    )
    parser.add_argument(
        "--context-len",
        type=int,
        default=10,
        help="Maximum previous utterances to include. Counselor and client lines each count as one utterance.",
    )
    args = parser.parse_args()

    dataset, input_files = read_dataset(args.input)
    examples, warnings = build_examples(
        dataset=dataset,
        context_len=args.context_len,
    )
    write_jsonl(args.output, examples)

    print(f"Input files: {len(input_files)}")
    for input_file in input_files:
        print(f"  - {input_file}")
    print(f"Read scenarios: {len(dataset)}")
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
