#!/usr/bin/env python3
"""Build panic add_dataset JSON from raw dialogue examples.

Pipeline:
1. Extract a panic-style client profile from each dialogue with GPT.
2. Retrieve the 3 most similar raw panic examples using SBERT.
3. Ask GPT to synthesize a new panic dataset sample from the profile and
   similar examples.
4. Save per-dialogue profile.json, similar_rows.json, sample.json, and the
   combined data/raw/add_dataset.json.

Similarity uses only:
- trigger
- physical_symptom
- catastrophic_thought
- emotional_react
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ADD_DIR = ROOT_DIR / "data/add_dataset"
DEFAULT_RAW_JSON = ROOT_DIR / "data/raw/panic_dataset.json"
DEFAULT_OUTPUT_JSON = ROOT_DIR / "data/raw/add_dataset.json"
DEFAULT_GENERATED_DIR = DEFAULT_ADD_DIR / "generated"

PROFILE_FIELDS = [
    "trigger",
    "physical_symptom",
    "catastrophic_thought",
    "emotional_react",
]

ALL_PROFILE_FIELDS = [
    "environment",
    "trigger",
    "physical_symptom",
    "emotional_react",
    "catastrophic_thought",
    "severity",
    "trigger_type",
]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def has_hangul(text: str) -> bool:
    return bool(re.search(r"[가-힣]", text))


def desired_language(dialogue: str) -> str:
    return "Korean" if has_hangul(dialogue) else "English"


def json_from_response_text(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def openai_json(client: OpenAI, *, model: str, messages: list[dict[str, Any]], temperature: float) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or "{}"
    return json_from_response_text(content)


def missing_fields(payload: dict[str, Any], fields: list[str]) -> list[str]:
    return [field for field in fields if not str(payload.get(field, "")).strip()]


def extract_profile(
    client: OpenAI,
    *,
    model: str,
    dialogue: str,
    output_language: str,
) -> dict[str, Any]:
    system = (
        "You create panic-attack counseling dataset metadata. "
        "Return only valid JSON. Do not diagnose; infer only dataset-style fields."
    )
    user = f"""
From the counseling dialogue, infer a panic-style client profile.

Return JSON with exactly these keys:
environment, trigger, physical_symptom, emotional_react,
catastrophic_thought, severity, trigger_type.

Field style:
- environment: where/when the panic episode is happening, if inferable.
- trigger: the main event or condition that started or worsened panic.
- physical_symptom: the main bodily sensations.
- emotional_react: the client's fear/anxiety response.
- catastrophic_thought: the feared catastrophic meaning, e.g. death, heart attack, losing control.
- severity: short label or phrase.
- trigger_type: one of physical, emotional, environmental, unknown when possible.

Language rule:
- If the source dialogue is English-only, write the profile in English.
- Otherwise, write inferred fields in Korean, except trigger_type may use the English labels above.

Output language: {output_language}

Dialogue:
{dialogue}
"""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    last_profile: dict[str, Any] = {}
    for _ in range(3):
        last_profile = openai_json(client, model=model, messages=messages, temperature=0.2)
        missing = missing_fields(last_profile, PROFILE_FIELDS)
        if not missing:
            return {field: str(last_profile.get(field, "")).strip() for field in ALL_PROFILE_FIELDS}
        messages = messages + [
            {"role": "assistant", "content": json.dumps(last_profile, ensure_ascii=False)},
            {
                "role": "user",
                "content": f"Missing required fields: {missing}. Return the complete JSON object.",
            },
        ]
    raise ValueError(f"Profile extraction failed; missing fields: {missing_fields(last_profile, PROFILE_FIELDS)}")


def profile_text(profile: dict[str, Any]) -> str:
    parts: list[str] = []
    for field in PROFILE_FIELDS:
        value = str(profile.get(field, "")).strip()
        if value:
            parts.append(f"{field}: {value}")
    return "\n".join(parts)


def dataset_fingerprint(path: Path) -> str:
    stat = path.stat()
    raw = f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{','.join(PROFILE_FIELDS)}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def read_panic_dataset(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected top-level JSON object, got {type(data).__name__}")
    return data


def flatten_dataset(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_id, sample in dataset.items():
        client = sample.get("client", {}) if isinstance(sample, dict) else {}
        dialogue = sample.get("dialogue", {}) if isinstance(sample, dict) else {}
        rows.append(
            {
                "sample_id": sample_id,
                "client": client,
                "plan": dialogue.get("plan", "") if isinstance(dialogue, dict) else "",
                "dialogue": dialogue.get("dialogue", []) if isinstance(dialogue, dict) else [],
            }
        )
    return rows


def retrieve_similar_sbert(
    rows: list[dict[str, Any]],
    query_profile: dict[str, Any],
    *,
    raw_json: Path,
    cache_dir: Path,
    top_k: int,
    model_name: str,
) -> list[dict[str, Any]]:
    from sentence_transformers import SentenceTransformer

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"panic_profile_sbert_{dataset_fingerprint(raw_json)}_{model_name.replace('/', '_')}.pkl"
    model = SentenceTransformer(model_name)
    if cache_path.exists():
        with cache_path.open("rb") as f:
            embeddings = pickle.load(f)["embeddings"]
    else:
        texts = [profile_text(row["client"]) for row in rows]
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        with cache_path.open("wb") as f:
            pickle.dump({"embeddings": embeddings, "fields": PROFILE_FIELDS}, f)

    query = model.encode([profile_text(query_profile)], normalize_embeddings=True)[0]
    scores = np.asarray(embeddings) @ np.asarray(query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [
        {
            "rank": rank + 1,
            "score": float(scores[index]),
            "row_index": int(index),
            **rows[int(index)],
        }
        for rank, index in enumerate(top_indices)
    ]


def retrieve_similar_tfidf(
    rows: list[dict[str, Any]],
    query_profile: dict[str, Any],
    *,
    raw_json: Path,
    cache_dir: Path,
    top_k: int,
) -> list[dict[str, Any]]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"panic_profile_tfidf_{dataset_fingerprint(raw_json)}.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as f:
            payload = pickle.load(f)
        vectorizer = payload["vectorizer"]
        matrix = payload["matrix"]
    else:
        texts = [profile_text(row["client"]) for row in rows]
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2)
        matrix = vectorizer.fit_transform(texts)
        with cache_path.open("wb") as f:
            pickle.dump({"vectorizer": vectorizer, "matrix": matrix, "fields": PROFILE_FIELDS}, f)

    query = vectorizer.transform([profile_text(query_profile)])
    scores = cosine_similarity(query, matrix)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [
        {
            "rank": rank + 1,
            "score": float(scores[index]),
            "row_index": int(index),
            **rows[int(index)],
        }
        for rank, index in enumerate(top_indices)
    ]


def retrieve_similar(
    rows: list[dict[str, Any]],
    query_profile: dict[str, Any],
    *,
    raw_json: Path,
    cache_dir: Path,
    top_k: int,
    backend: str,
    sbert_model: str,
) -> tuple[str, list[dict[str, Any]]]:
    if backend in {"auto", "sbert"}:
        try:
            return "sbert", retrieve_similar_sbert(
                rows,
                query_profile,
                raw_json=raw_json,
                cache_dir=cache_dir,
                top_k=top_k,
                model_name=sbert_model,
            )
        except ModuleNotFoundError:
            if backend == "sbert":
                raise
    return "tfidf", retrieve_similar_tfidf(
        rows,
        query_profile,
        raw_json=raw_json,
        cache_dir=cache_dir,
        top_k=top_k,
    )


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    client = row.get("client", {})
    return {
        "rank": row["rank"],
        "score": row["score"],
        "row_index": row["row_index"],
        "sample_id": row.get("sample_id", ""),
        "profile": {field: client.get(field, "") for field in PROFILE_FIELDS},
        "environment": client.get("environment", ""),
        "severity": client.get("severity", ""),
        "trigger_type": client.get("trigger_type", ""),
        "plan": row.get("plan", ""),
        "dialogue": row.get("dialogue", []),
    }


def parse_dialogue_text(dialogue_text: str) -> list[dict[str, Any]]:
    utterances: list[tuple[str, str]] = []
    for raw_line in dialogue_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.match(r"^(turn|턴)\s*\d+\s*:?\s*$", line, flags=re.IGNORECASE):
            continue
        line = re.sub(r"^\s*(?:turn\s*)?\d+\s*[\).:-]\s*", "", line, flags=re.IGNORECASE)
        match = re.match(r"^(Counselor|Client|상담사|내담자)\s*(?:\(\d+\))?\s*:\s*(.+)$", line)
        if not match:
            continue
        speaker = match.group(1)
        text = match.group(2).strip()
        role = "counselor" if speaker in {"Counselor", "상담사"} else "client"
        utterances.append((role, text))

    turns: list[dict[str, Any]] = []
    pending_counselor = ""
    for role, text in utterances:
        if role == "counselor":
            pending_counselor = text
            continue
        if role == "client" and pending_counselor:
            turns.append(
                {
                    "turn": len(turns) + 1,
                    "counselor": pending_counselor,
                    "client": text,
                }
            )
            pending_counselor = ""
    return turns


def synthesize_sample(
    client: OpenAI,
    *,
    model: str,
    profile: dict[str, Any],
    similar_rows: list[dict[str, Any]],
    output_language: str,
) -> dict[str, Any]:
    system = (
        "You generate one synthetic panic-attack emergency counseling plan and dialogue text. "
        "Return only valid JSON."
    )
    examples = [compact_row(row) for row in similar_rows]
    user = f"""
Create one new panic counseling plan and dialogue text.

Return JSON with exactly this structure:
{{
  "plan": "...",
  "dialogue_text": "Turn 1\\nCounselor: ...\\nClient: ...\\nTurn 2\\nCounselor: ...\\nClient: ..."
}}

You must combine:
1. The new profile.
2. The counseling flow and structure of the 3 most similar existing examples.

Hard requirements:
- Make a NEW dialogue, not a line-by-line copy of the similar examples.
- Keep the profile consistent with the generated dialogue.
- dialogue_text should have about 15 counselor-client pairs.
- Write dialogue_text as plain text, not a list and not dictionaries.
- Each pair must include the turn number as "Turn N" followed by exactly one
  "Counselor: ..." line and one "Client: ..." line.
- Follow the panic support flow:
  empathy and validation first;
  check current location/safety;
  guide to a quieter/less crowded/safer place when relevant;
  stabilize with breathing or grounding;
  when the client fears death/heart attack/losing control, empathize and calmly explain it can be a body alarm response without making a diagnosis;
  recommend 119, ER, nearby help, or professional care for severe breathing difficulty, chest pain, loss of consciousness, self-harm/other-harm risk, worsening symptoms, or persistent symptoms;
  when stable, summarize, encourage, and recommend professional help if symptoms continue or repeat.
- Do not mention this is generated.
- Use Korean if output language is Korean.

Output language: {output_language}

New profile:
{json.dumps(profile, ensure_ascii=False, indent=2)}

Similar existing examples:
{json.dumps(examples, ensure_ascii=False, indent=2)}
"""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    last_error = ""
    for _ in range(3):
        payload = openai_json(client, model=model, messages=messages, temperature=0.7)
        try:
            sample = normalize_sample(payload, profile=profile)
            validate_sample(sample, output_language=output_language)
            return sample
        except ValueError as exc:
            last_error = str(exc)
            messages = messages + [
                {"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)},
                {
                    "role": "user",
                    "content": (
                        f"The previous JSON is invalid: {last_error}. "
                        "Return a corrected complete JSON object with only plan and dialogue_text. "
                        "The plan must be a meaningful Korean counseling plan when output language is Korean."
                    ),
                },
            ]
    raise ValueError(f"Generated sample failed validation: {last_error}")


def normalize_sample(sample: dict[str, Any], *, profile: dict[str, Any]) -> dict[str, Any]:
    client_profile = {
        field: str(profile.get(field, "")).strip()
        for field in ALL_PROFILE_FIELDS
    }
    if client_profile["trigger_type"] not in {"physical", "emotional", "environmental", "unknown"}:
        trigger_type = client_profile["trigger_type"].lower()
        if "physical" in trigger_type or "신체" in trigger_type:
            client_profile["trigger_type"] = "physical"
        elif "emotion" in trigger_type or "감정" in trigger_type or "정서" in trigger_type:
            client_profile["trigger_type"] = "emotional"
        elif "environment" in trigger_type or "환경" in trigger_type:
            client_profile["trigger_type"] = "environmental"
        else:
            client_profile["trigger_type"] = "unknown"

    plan = str(sample.get("plan", "")).strip()
    dialogue_text = str(sample.get("dialogue_text", "")).strip()
    turns = parse_dialogue_text(dialogue_text)
    if not turns:
        raise ValueError("Generated dialogue_text has no valid counselor-client turns")
    return {
        "client": client_profile,
        "dialogue": {
            "plan": plan,
            "dialogue": turns,
        },
    }


def validate_sample(sample: dict[str, Any], *, output_language: str) -> None:
    plan = str(sample.get("dialogue", {}).get("plan", "")).strip()
    if len(plan) < 20:
        raise ValueError("dialogue.plan is too short")
    if plan.lower() in {"this is plan", "plan", "none", "n/a"}:
        raise ValueError("dialogue.plan is a placeholder")
    if output_language == "Korean" and not has_hangul(plan):
        raise ValueError("dialogue.plan must be Korean")
    turns = sample.get("dialogue", {}).get("dialogue", [])
    if not isinstance(turns, list) or not 14 <= len(turns) <= 16:
        raise ValueError("dialogue.dialogue must contain about 15 turns, accepted range is 14-16")
    for index, turn in enumerate(turns, start=1):
        if not isinstance(turn, dict):
            raise ValueError(f"turn {index} is not an object")
        if set(turn) != {"turn", "counselor", "client"}:
            raise ValueError(f"turn {index} must have exactly turn, counselor, client keys")
        if turn.get("turn") != index:
            raise ValueError(f"turn {index} has incorrect turn number")
        if not str(turn.get("counselor", "")).strip() or not str(turn.get("client", "")).strip():
            raise ValueError(f"turn {index} must include both counselor and client text")


def make_sample_id(stem: str, existing_ids: set[str]) -> str:
    safe_stem = re.sub(r"[^A-Za-z0-9가-힣_.-]+", "-", stem).strip("-") or "example"
    base = f"add-{safe_stem}"
    candidate = base
    suffix = 2
    while candidate in existing_ids:
        candidate = f"{base}-{suffix}"
        suffix += 1
    existing_ids.add(candidate)
    return candidate


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def collect_dialogues(dialogue_dir: Path, dialogue: Path | None) -> list[tuple[str, Path]]:
    if dialogue:
        return [(dialogue.stem, dialogue)]
    dialogue_paths = sorted(dialogue_dir.glob("*.txt"))
    if not dialogue_paths:
        raise FileNotFoundError(f"No dialogue .txt files found in {dialogue_dir}")
    return [(path.stem, path) for path in dialogue_paths]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dialogue-dir", type=Path, default=DEFAULT_ADD_DIR / "raw/dialogue")
    parser.add_argument("--dialogue", type=Path, default=None, help="Process one dialogue file instead of all dialogue/*.txt files.")
    parser.add_argument("--raw-json", type=Path, default=DEFAULT_RAW_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--generated-dir", type=Path, default=DEFAULT_GENERATED_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_ADD_DIR / "cache")
    parser.add_argument("--profile-model", default="gpt-4o")
    parser.add_argument("--dialogue-model", default="gpt-4o")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--embedding-backend", choices=["auto", "sbert", "tfidf"], default="sbert")
    parser.add_argument("--sbert-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--profile-json", type=Path, default=None, help="Reuse an existing profile JSON. Only valid with --dialogue.")
    parser.add_argument("--retrieve-only", action="store_true", help="Stop after profile extraction and similar-row retrieval.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.profile_json and not args.dialogue:
        raise ValueError("--profile-json can only be used with --dialogue")

    dataset = read_panic_dataset(args.raw_json)
    rows = flatten_dataset(dataset)
    examples = collect_dialogues(args.dialogue_dir, args.dialogue)
    client: OpenAI | None = None
    backend_used = ""
    output_samples: dict[str, Any] = {}
    existing_ids = set(dataset)

    for stem, dialogue_path in examples:
        print(f"Processing {stem}: dialogue={dialogue_path}", flush=True)
        dialogue = read_text(dialogue_path)
        output_language = desired_language(dialogue)
        item_generated_dir = args.generated_dir / stem

        if args.profile_json:
            profile = json.loads(args.profile_json.read_text(encoding="utf-8"))
        else:
            if client is None:
                client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            profile = extract_profile(
                client,
                model=args.profile_model,
                dialogue=dialogue,
                output_language=output_language,
            )
        write_json(item_generated_dir / "profile.json", profile)

        backend_used, similar_rows = retrieve_similar(
            rows,
            profile,
            raw_json=args.raw_json,
            cache_dir=args.cache_dir,
            top_k=args.top_k,
            backend=args.embedding_backend,
            sbert_model=args.sbert_model,
        )
        write_json(
            item_generated_dir / "similar_rows.json",
            {
                "source_dialogue": str(dialogue_path),
                "backend": backend_used,
                "match_fields": PROFILE_FIELDS,
                "query_profile": {field: profile.get(field, "") for field in PROFILE_FIELDS},
                "rows": [compact_row(row) for row in similar_rows],
            },
        )
        if not args.retrieve_only:
            if client is None:
                client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            output_sample = synthesize_sample(
                client,
                model=args.dialogue_model,
                profile=profile,
                similar_rows=similar_rows,
                output_language=output_language,
            )
            sample_id = make_sample_id(stem, existing_ids)
            output_samples[sample_id] = output_sample
            write_json(item_generated_dir / "sample.json", {sample_id: output_sample})
        print(f"  Profile: {item_generated_dir / 'profile.json'}", flush=True)
        print(f"  Similar rows: {item_generated_dir / 'similar_rows.json'}", flush=True)
        if not args.retrieve_only:
            print(f"  Sample: {item_generated_dir / 'sample.json'}", flush=True)

    print(f"Processed examples: {len(examples)}")
    print(f"Embedding backend: {backend_used}")
    if not args.retrieve_only:
        write_json(args.output_json, output_samples)
        print(f"Wrote samples: {len(output_samples)}")
        print(f"Output JSON: {args.output_json}")


if __name__ == "__main__":
    main()
