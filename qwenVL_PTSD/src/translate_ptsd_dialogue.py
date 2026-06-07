#!/usr/bin/env python3
"""Translate selected PTSD dialogue fields to Korean with OpenAI Batch API.

The source PTSD files contain many metadata and prompt/debug fields. This script
keeps the full JSON structure unchanged and translates only:

    dialogue[*].counselor.revised_turn_strategy
    dialogue[*].counselor.response
    dialogue[*].client.response

Workflow:
    inspect   local summary, no API calls
    prepare   create Batch API JSONL input, no API calls
    submit    upload JSONL and create batch jobs
    status    check batch job status
    download  download completed outputs
    apply     write translated JSON files
    run       prepare, submit, wait, download, apply
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_INPUT_DIR = Path("/home/jihyunlee/mirror/MIRROR_code/data/PTSD")
DEFAULT_OUTPUT_DIR = Path("/home/jihyunlee/mirror/MIRROR_code/data/PTSD_ko")
DEFAULT_WORK_DIR = Path("/home/jihyunlee/mirror/translate/ptsd_dialogue_ko_batch")
DEFAULT_API_CONFIG = Path("/home/jihyunlee/mirror/configs/api.json")
DEFAULT_API_KEY_FIELD = "api-key"
DEFAULT_MODEL = "gpt-4o-mini"
COUNSELOR_TARGET_FIELDS = ("revised_turn_strategy", "response")
CLIENT_TARGET_FIELDS = ("response",)


SYSTEM_PROMPT = """You are a careful Korean translation engine for a PTSD counseling dataset.
Translate only the provided string values from English to natural Korean.
Preserve all JSON keys exactly.
Preserve clinical meaning, CBT counseling nuance, strategy labels inside the text, numeric values, IDs, and [END] markers.
Keep counselor and client responses concise and natural in Korean.
Do not add explanations, notes, or extra keys.
Return only a valid JSON object with the same keys as the input object."""


def read_api_key(config_path: Path, key_field: str) -> str:
    with config_path.open(encoding="utf-8") as f:
        config = json.load(f)
    api_key = config.get(key_field)
    if not api_key:
        available = ", ".join(sorted(config.keys()))
        raise ValueError(
            f"API key field '{key_field}' was not found in {config_path}. "
            f"Available fields: {available}"
        )
    return api_key


def make_client(config_path: Path, key_field: str) -> Any:
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The openai package is required. Run this in the conda/env where OpenAI SDK is installed."
        ) from exc
    return OpenAI(api_key=read_api_key(config_path, key_field))


def source_files(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.glob("*.json") if path.is_file())


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path} has non-object top-level JSON")
    return data


def iter_translation_items(
    input_dir: Path,
    num_files: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    files = source_files(input_dir)
    if num_files != -1:
        if num_files < 0:
            raise ValueError("--num-files must be positive or -1 for all files")
        files = files[:num_files]

    items: list[dict[str, Any]] = []
    warnings: list[str] = []
    for file_index, path in enumerate(files):
        data = read_json(path)
        dialogue = data.get("dialogue")
        if not isinstance(dialogue, list):
            warnings.append(f"{path.name}: dialogue is not a list")
            continue

        for turn_index, turn in enumerate(dialogue):
            counselor = turn.get("counselor") if isinstance(turn, dict) else None
            if not isinstance(counselor, dict):
                warnings.append(f"{path.name}: turn {turn_index + 1} counselor is missing")
                continue

            payload = {
                f"counselor.{field}": counselor.get(field, "")
                for field in COUNSELOR_TARGET_FIELDS
                if isinstance(counselor.get(field, ""), str) and counselor.get(field, "").strip()
            }

            client = turn.get("client") if isinstance(turn, dict) else None
            if isinstance(client, dict):
                payload.update(
                    {
                        f"client.{field}": client.get(field, "")
                        for field in CLIENT_TARGET_FIELDS
                        if isinstance(client.get(field, ""), str) and client.get(field, "").strip()
                    }
                )
            else:
                warnings.append(f"{path.name}: turn {turn_index + 1} client is missing")

            if not payload:
                continue

            items.append(
                {
                    "custom_id": f"file-{file_index}-turn-{turn_index}",
                    "file_name": path.name,
                    "turn_index": turn_index,
                    "payload": payload,
                }
            )
    return items, warnings


def build_request(item: dict[str, Any], model: str) -> dict[str, Any]:
    return {
        "custom_id": item["custom_id"],
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Translate this JSON object's string values to Korean. "
                        "Return a JSON object with exactly the same keys.\n"
                        f"{json.dumps(item['payload'], ensure_ascii=False)}"
                    ),
                },
            ],
        },
    }


def write_jsonl_shards(
    input_dir: Path,
    work_dir: Path,
    model: str,
    max_requests_per_file: int,
    num_files: int,
) -> list[Path]:
    if max_requests_per_file <= 0:
        raise ValueError("--max-requests-per-file must be positive")

    items, warnings = iter_translation_items(input_dir=input_dir, num_files=num_files)
    work_dir.mkdir(parents=True, exist_ok=True)
    request_paths: list[Path] = []
    current_file = None

    try:
        for item_index, item in enumerate(items):
            if item_index % max_requests_per_file == 0:
                if current_file:
                    current_file.close()
                shard_no = len(request_paths)
                current_path = work_dir / f"batch_input_{shard_no:03d}.jsonl"
                current_file = current_path.open("w", encoding="utf-8")
                request_paths.append(current_path)

            assert current_file is not None
            current_file.write(json.dumps(build_request(item, model), ensure_ascii=False) + "\n")
    finally:
        if current_file:
            current_file.close()

    manifest = {
        "input_dir": str(input_dir),
        "target_fields": {
            "counselor": list(COUNSELOR_TARGET_FIELDS),
            "client": list(CLIENT_TARGET_FIELDS),
        },
        "model": model,
        "num_files": num_files,
        "source_files": [path.name for path in source_files(input_dir)[: None if num_files == -1 else num_files]],
        "request_count": len(items),
        "created_at": int(time.time()),
        "request_files": [str(path) for path in request_paths],
        "items": [
            {
                "custom_id": item["custom_id"],
                "file_name": item["file_name"],
                "turn_index": item["turn_index"],
                "fields": sorted(item["payload"].keys()),
            }
            for item in items
        ],
        "warnings": warnings,
    }
    (work_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return request_paths


def submit_batches(client: Any, work_dir: Path) -> list[dict[str, Any]]:
    manifest_path = work_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Run prepare first. Missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs: list[dict[str, Any]] = []

    for request_file in manifest["request_files"]:
        path = Path(request_file)
        with path.open("rb") as f:
            uploaded = client.files.create(file=f, purpose="batch")
        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"source": "ptsd_dialogue_ko", "input_file": path.name},
        )
        jobs.append(
            {
                "request_file": str(path),
                "input_file_id": uploaded.id,
                "batch_id": batch.id,
                "status": batch.status,
            }
        )
        print(f"submitted {path.name}: file={uploaded.id} batch={batch.id} status={batch.status}")

    jobs_path = work_dir / "batch_jobs.json"
    jobs_path.write_text(json.dumps(jobs, indent=2), encoding="utf-8")
    return jobs


def load_jobs(work_dir: Path) -> list[dict[str, Any]]:
    jobs_path = work_dir / "batch_jobs.json"
    if not jobs_path.exists():
        raise FileNotFoundError(f"Run submit first. Missing {jobs_path}")
    return json.loads(jobs_path.read_text(encoding="utf-8"))


def format_batch_status(batch: Any) -> str:
    counts = batch.request_counts
    count_text = ""
    if counts:
        count_text = f" total={counts.total} completed={counts.completed} failed={counts.failed}"
    return (
        f"{batch.id}: status={batch.status}{count_text} "
        f"output_file_id={batch.output_file_id} error_file_id={batch.error_file_id}"
    )


def show_status(client: Any, work_dir: Path) -> None:
    for job in load_jobs(work_dir):
        print(format_batch_status(client.batches.retrieve(job["batch_id"])))


def wait_for_batches(client: Any, work_dir: Path, poll_seconds: int) -> None:
    terminal_statuses = {"completed", "failed", "expired", "cancelled"}
    if poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")

    while True:
        batches = [client.batches.retrieve(job["batch_id"]) for job in load_jobs(work_dir)]
        for batch in batches:
            print(format_batch_status(batch), flush=True)

        statuses = {batch.status for batch in batches}
        if statuses.issubset(terminal_statuses):
            bad_statuses = statuses - {"completed"}
            if bad_statuses:
                raise RuntimeError(f"Some batch jobs did not complete successfully: {sorted(bad_statuses)}")
            return

        print(f"waiting {poll_seconds}s before next status check...", flush=True)
        time.sleep(poll_seconds)


def write_file_content(client: Any, file_id: str, output_path: Path) -> None:
    content = client.files.content(file_id)
    if hasattr(content, "write_to_file"):
        content.write_to_file(str(output_path))
        return
    data = content.read() if hasattr(content, "read") else bytes(content)
    output_path.write_bytes(data)


def download_outputs(client: Any, work_dir: Path) -> None:
    output_dir = work_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    updated_jobs = []

    for job in load_jobs(work_dir):
        batch = client.batches.retrieve(job["batch_id"])
        job.update(
            {
                "status": batch.status,
                "output_file_id": batch.output_file_id,
                "error_file_id": batch.error_file_id,
            }
        )
        if batch.output_file_id:
            out_path = output_dir / f"{batch.id}_output.jsonl"
            write_file_content(client, batch.output_file_id, out_path)
            job["output_path"] = str(out_path)
            print(f"downloaded output: {out_path}")
        if batch.error_file_id:
            err_path = output_dir / f"{batch.id}_errors.jsonl"
            write_file_content(client, batch.error_file_id, err_path)
            job["error_path"] = str(err_path)
            print(f"downloaded errors: {err_path}")
        updated_jobs.append(job)

    (work_dir / "batch_jobs.json").write_text(json.dumps(updated_jobs, indent=2), encoding="utf-8")


def parse_translation_from_response(line: str) -> tuple[str, dict[str, str]]:
    item = json.loads(line)
    custom_id = item["custom_id"]
    if item.get("error"):
        raise ValueError(f"{custom_id} failed: {item['error']}")
    body = item["response"]["body"]
    content = body["choices"][0]["message"]["content"]
    translated = json.loads(content)
    if not isinstance(translated, dict):
        raise TypeError(f"{custom_id} returned non-object translation")
    return custom_id, {key: str(value) for key, value in translated.items()}


def load_translations(work_dir: Path) -> dict[str, dict[str, str]]:
    translations: dict[str, dict[str, str]] = {}
    output_dir = work_dir / "outputs"
    for path in sorted(output_dir.glob("*_output.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                custom_id, translated = parse_translation_from_response(line)
                translations[custom_id] = translated
    if not translations:
        raise ValueError(f"No translations found in {output_dir}")
    return translations


def apply_translations(input_dir: Path, work_dir: Path, output_dir: Path) -> None:
    manifest_path = work_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Run prepare first. Missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    item_map = {
        item["custom_id"]: item
        for item in manifest.get("items", [])
    }
    translations = load_translations(work_dir)
    by_file: dict[str, dict[int, dict[str, str]]] = {}

    for custom_id, translated in translations.items():
        item = item_map.get(custom_id)
        if not item:
            raise KeyError(f"{custom_id} is not present in manifest")
        file_name = item["file_name"]
        turn_index = int(item["turn_index"])
        by_file.setdefault(file_name, {})[turn_index] = translated

    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    updated_turns = 0
    for file_name in manifest["source_files"]:
        source_path = input_dir / file_name
        data = copy.deepcopy(read_json(source_path))
        dialogue = data.get("dialogue", [])
        for turn_index, translated in by_file.get(file_name, {}).items():
            counselor = dialogue[turn_index]["counselor"]
            client = dialogue[turn_index].get("client")
            for field in COUNSELOR_TARGET_FIELDS:
                key = f"counselor.{field}"
                if key in translated:
                    counselor[field] = translated[key]
            if isinstance(client, dict):
                for field in CLIENT_TARGET_FIELDS:
                    key = f"client.{field}"
                    if key in translated:
                        client[field] = translated[key]
            updated_turns += 1

        out_path = output_dir / file_name
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written += 1

    print(f"wrote files: {written}")
    print(f"updated dialogue turns: {updated_turns}")
    print(f"output_dir: {output_dir}")


def inspect_dataset(input_dir: Path, sample_files: int) -> None:
    files = source_files(input_dir)
    items, warnings = iter_translation_items(input_dir=input_dir, num_files=-1)
    print(f"json files: {len(files)}")
    print(f"translation requests: {len(items)}")
    print(f"counselor target fields: {', '.join(COUNSELOR_TARGET_FIELDS)}")
    print(f"client target fields: {', '.join(CLIENT_TARGET_FIELDS)}")
    print(f"warnings: {len(warnings)}")
    for warning in warnings[:20]:
        print(f"  - {warning}")

    for path in files[:sample_files]:
        data = read_json(path)
        first_turn = data["dialogue"][0]
        sample = {
            "counselor": {
                field: first_turn["counselor"].get(field, "")
                for field in COUNSELOR_TARGET_FIELDS
            },
            "client": {
                field: first_turn.get("client", {}).get(field, "")
                for field in CLIENT_TARGET_FIELDS
            },
        }
        print(f"\n{path.name}")
        print(json.dumps(sample, ensure_ascii=False, indent=2)[:2000])


def run_pipeline(args: argparse.Namespace) -> None:
    paths = write_jsonl_shards(
        input_dir=args.input_dir,
        work_dir=args.work_dir,
        model=args.model,
        max_requests_per_file=args.max_requests_per_file,
        num_files=args.num_files,
    )
    print(f"created {len(paths)} JSONL file(s) in {args.work_dir}")

    client = make_client(args.api_config, args.api_key_field)
    submit_batches(client, args.work_dir)
    wait_for_batches(client, args.work_dir, args.poll_seconds)
    download_outputs(client, args.work_dir)
    apply_translations(args.input_dir, args.work_dir, args.output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--api-config", type=Path, default=DEFAULT_API_CONFIG)
    parser.add_argument("--api-key-field", default=DEFAULT_API_KEY_FIELD)

    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--sample-files", type=int, default=1)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--model", default=DEFAULT_MODEL)
    prepare.add_argument("--max-requests-per-file", type=int, default=45_000)
    prepare.add_argument(
        "--num-files",
        "--limit",
        dest="num_files",
        type=int,
        default=10,
        help="Number of JSON files to include. Use -1 for all files. Default: 10.",
    )

    subparsers.add_parser("submit")
    subparsers.add_parser("status")
    subparsers.add_parser("download")
    subparsers.add_parser("apply")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--model", default=DEFAULT_MODEL)
    run_parser.add_argument("--max-requests-per-file", type=int, default=45_000)
    run_parser.add_argument(
        "--num-files",
        "--limit",
        dest="num_files",
        type=int,
        default=10,
        help="Number of JSON files to include. Use -1 for all files. Default: 10.",
    )
    run_parser.add_argument("--poll-seconds", type=int, default=60)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "inspect":
            inspect_dataset(args.input_dir, args.sample_files)
            return 0
        if args.command == "prepare":
            paths = write_jsonl_shards(
                input_dir=args.input_dir,
                work_dir=args.work_dir,
                model=args.model,
                max_requests_per_file=args.max_requests_per_file,
                num_files=args.num_files,
            )
            print(f"created {len(paths)} JSONL file(s) in {args.work_dir}")
            return 0
        if args.command == "apply":
            apply_translations(args.input_dir, args.work_dir, args.output_dir)
            return 0
        if args.command == "run":
            run_pipeline(args)
            return 0

        client = make_client(args.api_config, args.api_key_field)
        if args.command == "submit":
            submit_batches(client, args.work_dir)
        elif args.command == "status":
            show_status(client, args.work_dir)
        elif args.command == "download":
            download_outputs(client, args.work_dir)
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
