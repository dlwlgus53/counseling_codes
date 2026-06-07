#!/usr/bin/env python3
"""Run add_dataset image synthesis: GPT expression prompts, PhotoMaker prompts, images."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[3]
IMAGE_SYNTHESIS_DIR = ROOT_DIR / "data/add_dataset/image_synthesis"


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=ROOT_DIR / "data/raw/add_dataset.csv")
    parser.add_argument("--save-dir", type=Path, default=ROOT_DIR / "data/mirror_images")
    parser.add_argument("--skip-gpt", action="store_true", help="Reuse existing GPT annotations.")
    parser.add_argument("--skip-photomaker", action="store_true", help="Stop after PhotoMaker prompt JSONL.")
    parser.add_argument("--gpt-model", default="gpt-4o", help="OpenAI model for image-expression annotation.")
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=19)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt_path = IMAGE_SYNTHESIS_DIR / "annot_data/gpt_prompt.jsonl"
    gpt_result_path = IMAGE_SYNTHESIS_DIR / "annot_data/gpt_result.jsonl"
    photomaker_prompt_path = IMAGE_SYNTHESIS_DIR / "photomaker_prompts/prompt.jsonl"

    run([
        sys.executable,
        str(IMAGE_SYNTHESIS_DIR / "preprocess_for_llm.py"),
        "--data-path",
        str(args.data_path),
        "--save-path",
        str(prompt_path),
    ])
    if not args.skip_gpt:
        run([
            sys.executable,
            str(IMAGE_SYNTHESIS_DIR / "annotate_gpt.py"),
            "--prompt-path",
            str(prompt_path),
            "--save-path",
            str(gpt_result_path),
            "--model",
            args.gpt_model,
        ])
    run([
        sys.executable,
        str(IMAGE_SYNTHESIS_DIR / "preprocess_for_photomaker.py"),
        "--data-path",
        str(args.data_path),
        "--gpt-result-path",
        str(gpt_result_path),
        "--save-path",
        str(photomaker_prompt_path),
    ])
    if args.skip_photomaker:
        print(f"PhotoMaker prompt JSONL: {photomaker_prompt_path}")
        return
    run([
        sys.executable,
        str(IMAGE_SYNTHESIS_DIR / "run_photomaker.py"),
        "--prompt_path",
        str(photomaker_prompt_path),
        "--save_dir",
        str(args.save_dir),
        "--num_steps",
        str(args.num_steps),
        "--seed",
        str(args.seed),
    ])


if __name__ == "__main__":
    main()
