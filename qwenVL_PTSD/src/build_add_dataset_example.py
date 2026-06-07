#!/usr/bin/env python3
"""Build PTSD add_dataset JSONL from raw dialogue examples.

Pipeline:
1. Extract a PTSD-style profile from each dialogue with GPT.
2. Retrieve the 3 most similar raw PTSD examples using SBERT or TF-IDF.
3. Ask GPT to synthesize a new PTSD counseling dialogue from the profile and
   similar examples.
4. Save per-dialogue profile.json, similar_rows.json, sample.json, and the
   combined data/raw/add_dataset.jsonl.
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
DEFAULT_RAW_INPUT = ROOT_DIR / "data/raw/PTSD.jsonl"
DEFAULT_OUTPUT_JSONL = ROOT_DIR / "data/raw/add_dataset.jsonl"
DEFAULT_GENERATED_DIR = DEFAULT_ADD_DIR / "generated"

PROFILE_FIELDS = [
    "daily_difficulty_ko",
    "belief_state_ko",
    "counseling_goal_ko",
]

PROFILE_REQUIRED_FIELDS = [
    "trauma_event",
    "daily_difficulty_ko",
    "daily_difficulty_en",
    "belief_state_ko",
    "belief_state_en",
    "counseling_goal_ko",
    "counseling_goal_en",
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


def missing_profile_fields(profile: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for field in PROFILE_REQUIRED_FIELDS:
        value = profile.get(field)
        if isinstance(value, dict):
            if not value:
                missing.append(field)
        elif not str(value or "").strip():
            missing.append(field)
    return missing


def text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " / ".join(f"{key}: {item}" for key, item in value.items() if str(item).strip())
    if isinstance(value, list):
        return " / ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def structured_profile_text(field: str, value: Any) -> str:
    if not isinstance(value, dict):
        return text_value(value)

    if field.endswith("_ko"):
        ko_keys = {
            "safety": "안전",
            "self": "자기인식",
            "others": "타인",
            "world": "세계관",
            "control": "통제감",
            "responsibility": "책임감",
            "future_danger": "미래위험",
        }
        if field.startswith("daily_difficulty"):
            return " / ".join(
                part
                for part in [
                    f"트리거: {value.get('trigger', '')}".strip(),
                    f"부적응적_사고: {value.get('maladaptive_thought', '')}".strip(),
                    f"회피행동: {value.get('avoidance_behavior', '')}".strip(),
                ]
                if not part.endswith(":")
            )
        if field.startswith("belief_state"):
            return "신념상태: " + " / ".join(
                f"{ko_keys.get(str(key), str(key))}: {item}"
                for key, item in value.items()
                if str(item).strip()
            )
        if field.startswith("counseling_goal"):
            return " / ".join(
                part
                for part in [
                    f"적응적_사고: {value.get('adaptive_thought', '')}".strip(),
                    f"적응적_행동: {value.get('adaptive_behavior', '')}".strip(),
                ]
                if not part.endswith(":")
            )

    if field.startswith("daily_difficulty"):
        return " / ".join(
            part
            for part in [
                f"Trigger: {value.get('trigger', '')}".strip(),
                f"Maladaptive thought: {value.get('maladaptive_thought', '')}".strip(),
                f"Avoidance behavior: {value.get('avoidance_behavior', '')}".strip(),
            ]
            if not part.endswith(":")
        )
    if field.startswith("belief_state"):
        return "Belief state: " + " / ".join(f"{key}: {item}" for key, item in value.items() if str(item).strip())
    if field.startswith("counseling_goal"):
        return " / ".join(
            part
            for part in [
                f"Adaptive thought: {value.get('adaptive_thought', '')}".strip(),
                f"Adaptive behavior: {value.get('adaptive_behavior', '')}".strip(),
            ]
            if not part.endswith(":")
        )
    return text_value(value)


def normalize_profile(profile: dict[str, Any]) -> dict[str, Any]:
    trauma_event = profile.get("trauma_event")
    if not isinstance(trauma_event, dict):
        trauma_event = {
            "event_description": str(trauma_event or "").strip(),
            "exposure_type": "",
            "perceived_threat_level": "",
            "time_since_event": "",
        }
    normalized = {
        "name": text_value(profile.get("name", "Synthetic client")) or "Synthetic client",
        "gender": text_value(profile.get("gender", "")),
        "age": text_value(profile.get("age", "")),
        "trauma_event": {
            "event_description": text_value(trauma_event.get("event_description", "")),
            "exposure_type": text_value(trauma_event.get("exposure_type", "")),
            "perceived_threat_level": text_value(trauma_event.get("perceived_threat_level", "")),
            "time_since_event": text_value(trauma_event.get("time_since_event", "")),
        },
    }
    for field in [
        "daily_difficulty_en",
        "daily_difficulty_ko",
        "belief_state_en",
        "belief_state_ko",
        "counseling_goal_en",
        "counseling_goal_ko",
    ]:
        normalized[field] = structured_profile_text(field, profile.get(field, ""))
    return normalized


def extract_profile(
    client: OpenAI,
    *,
    model: str,
    dialogue: str,
    output_language: str,
) -> dict[str, Any]:
    system = (
        "You create trauma-informed CBT counseling dataset metadata for PTSD. "
        "Return only valid JSON. Do not diagnose; infer dataset-style fields."
    )
    user = f"""
From the counseling dialogue, infer a PTSD-style client profile compatible with the existing dataset.

Return JSON with exactly these keys:
name, gender, age, trauma_event, daily_difficulty_en, daily_difficulty_ko,
belief_state_en, belief_state_ko, counseling_goal_en, counseling_goal_ko.

trauma_event must be an object with:
event_description, exposure_type, perceived_threat_level, time_since_event.

Field definitions:
- daily_difficulty_* must include trigger, maladaptive thought, and avoidance behavior.
- belief_state_* must describe a broader post-trauma belief/schema about safety, self,
  other people, control, responsibility, or future danger.
- counseling_goal_* must include adaptive thought and adaptive behavior.
- Make clinically plausible inferences without overclaiming.
- Do not blame the client or minimize the traumatic event.
- If the source dialogue is Korean, write Korean fields naturally in Korean.
- English and Korean fields should express the same meaning.

Output language for dialogue-derived fields: {output_language}

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
        profile = normalize_profile(last_profile)
        missing = missing_profile_fields(profile)
        if not missing:
            return profile
        messages = messages + [
            {"role": "assistant", "content": json.dumps(last_profile, ensure_ascii=False)},
            {
                "role": "user",
                "content": f"Missing required fields: {missing}. Return the complete JSON object.",
            },
        ]
    raise ValueError(f"Profile extraction failed; missing fields: {missing_profile_fields(normalize_profile(last_profile))}")


def profile_text(profile: dict[str, Any]) -> str:
    parts: list[str] = []
    trauma_event = profile.get("trauma_event", {})
    if isinstance(trauma_event, dict):
        event_description = str(trauma_event.get("event_description", "")).strip()
        if event_description:
            parts.append(f"trauma_event: {event_description}")
    for field in PROFILE_FIELDS:
        value = str(profile.get(field, "")).strip()
        if value:
            parts.append(f"{field}: {value}")
    return "\n".join(parts)


def dataset_fingerprint(path: Path) -> str:
    stat = path.stat()
    raw = f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{','.join(PROFILE_FIELDS)}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def read_jsonl_dataset(input_path: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    if input_path.is_file():
        paths = [input_path]
    elif input_path.is_dir():
        paths = sorted(path for path in input_path.glob("*.jsonl") if path.name != "add_dataset.jsonl")
        if not paths:
            raise FileNotFoundError(f"No source .jsonl files found in {input_path}")
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if isinstance(record, dict):
                    records.append(record)
    return records, paths


def flatten_dataset(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        profile = record.get("profile", {})
        if not isinstance(profile, dict):
            profile = {}
        rows.append(
            {
                "news_id": record.get("news_id", ""),
                "description": record.get("description", ""),
                "language": record.get("language", ""),
                "profile": profile,
                "dialogue": record.get("dialogue", []),
            }
        )
    return rows


def retrieve_similar_sbert(
    rows: list[dict[str, Any]],
    query_profile: dict[str, Any],
    *,
    raw_input: Path,
    cache_dir: Path,
    top_k: int,
    model_name: str,
) -> list[dict[str, Any]]:
    from sentence_transformers import SentenceTransformer

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"ptsd_profile_sbert_{dataset_fingerprint(raw_input)}_{model_name.replace('/', '_')}.pkl"
    model = SentenceTransformer(model_name)
    if cache_path.exists():
        with cache_path.open("rb") as f:
            embeddings = pickle.load(f)["embeddings"]
    else:
        texts = [profile_text(row["profile"]) for row in rows]
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
    raw_input: Path,
    cache_dir: Path,
    top_k: int,
) -> list[dict[str, Any]]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"ptsd_profile_tfidf_{dataset_fingerprint(raw_input)}.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as f:
            payload = pickle.load(f)
        vectorizer = payload["vectorizer"]
        matrix = payload["matrix"]
    else:
        texts = [profile_text(row["profile"]) for row in rows]
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
    raw_input: Path,
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
                raw_input=raw_input,
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
        raw_input=raw_input,
        cache_dir=cache_dir,
        top_k=top_k,
    )


def compact_dialogue(dialogue: Any, max_turns: int = 32) -> list[dict[str, str]]:
    if not isinstance(dialogue, list):
        return []
    compact: list[dict[str, str]] = []
    for turn in dialogue[:max_turns]:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role", "")).strip()
        utterance = str(turn.get("utterance", "")).strip()
        if role and utterance:
            compact.append({"role": role, "utterance": utterance})
    return compact


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    profile = row.get("profile", {})
    return {
        "rank": row["rank"],
        "score": row["score"],
        "row_index": row["row_index"],
        "news_id": row.get("news_id", ""),
        "profile": {
            "trauma_event": profile.get("trauma_event", {}) if isinstance(profile, dict) else {},
            **({field: profile.get(field, "") for field in PROFILE_FIELDS} if isinstance(profile, dict) else {}),
        },
        "dialogue": compact_dialogue(row.get("dialogue", [])),
    }


def parse_dialogue_text(dialogue_text: str) -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    for raw_line in dialogue_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.match(r"^(turn|턴)\s*\d+\s*:?\s*$", line, flags=re.IGNORECASE):
            continue
        line = re.sub(r"^\s*(?:turn\s*)?\d+\s*[\).:-]\s*", "", line, flags=re.IGNORECASE)
        match = re.match(r"^(Counselor|Client|Therapist|상담자|상담사|내담자)\s*(?:\(\d+\))?\s*:\s*(.+)$", line)
        if not match:
            continue
        speaker = match.group(1)
        text = match.group(2).strip()
        role = "client" if speaker in {"Client", "내담자"} else "counselor"
        turns.append({"role": role, "utterance": text})
    return turns


def synthesize_record(
    client: OpenAI,
    *,
    model: str,
    profile: dict[str, Any],
    similar_rows: list[dict[str, Any]],
    output_language: str,
) -> dict[str, Any]:
    system = (
        "You generate one synthetic trauma-informed CBT counseling dialogue for a PTSD dataset. "
        "Return only valid JSON."
    )
    examples = [compact_row(row) for row in similar_rows]
    user = f"""
Create one new PTSD counseling dialogue.

Return JSON with exactly this structure:
{{
  "description": "...",
  "dialogue_text": "Counselor: ...\\nClient: ...\\nCounselor: ...\\nClient: ..."
}}

You must combine:
1. The new profile.
2. The counseling flow and style of the 3 most similar existing examples.

Hard requirements:
- Make a NEW dialogue, not a line-by-line copy of the similar examples.
- Keep the profile consistent with the generated dialogue.
- dialogue_text must be plain text, not a list and not dictionaries.
- Use exactly one speaker line per utterance.
- Use speaker labels "Counselor:" and "Client:".
- Start with a Counselor line and end with a Client line.
- Generate 24 to 40 total utterances.
- The counselor should first invite the client to talk about recent daily difficulty,
  not directly reveal the trauma, profile, belief state, or counseling goal.
- Gradually explore trigger, avoidance behavior, maladaptive thought, broader
  post-trauma belief state, and the counseling goal.
- Use empathy, validation, Socratic questioning, evidence checking,
  decatastrophizing, and past-present differentiation.
- Guide the client toward the adaptive thought gradually.
- End with one small, safe, realistic practice task.
- Do not recommend sudden or intense exposure.
- Do not include diagnosis, medication advice, or forced positivity.
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
            record = normalize_record(payload, profile=profile)
            validate_record(record, output_language=output_language)
            return record
        except ValueError as exc:
            last_error = str(exc)
            messages = messages + [
                {"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)},
                {
                    "role": "user",
                    "content": (
                        f"The previous JSON is invalid: {last_error}. "
                        "Return a corrected complete JSON object with only description and dialogue_text."
                    ),
                },
            ]
    raise ValueError(f"Generated record failed validation: {last_error}")


def normalize_record(payload: dict[str, Any], *, profile: dict[str, Any]) -> dict[str, Any]:
    description = str(payload.get("description", "")).strip()
    dialogue_text = str(payload.get("dialogue_text", "")).strip()
    dialogue = parse_dialogue_text(dialogue_text)
    if not dialogue:
        raise ValueError("Generated dialogue_text has no valid speaker lines")
    return {
        "language": "ko" if has_hangul(dialogue_text) else "en",
        "description": description,
        "profile": normalize_profile(profile),
        "dialogue": dialogue,
    }


def validate_record(record: dict[str, Any], *, output_language: str) -> None:
    dialogue = record.get("dialogue", [])
    if not isinstance(dialogue, list) or not 24 <= len(dialogue) <= 40:
        raise ValueError("dialogue must contain 24 to 40 utterances")
    if len(dialogue) % 2 != 0:
        raise ValueError("dialogue must contain an even number of utterances")
    if dialogue[0].get("role") != "counselor":
        raise ValueError("dialogue must start with counselor")
    if dialogue[-1].get("role") != "client":
        raise ValueError("dialogue must end with client")
    for index, turn in enumerate(dialogue):
        expected_role = "counselor" if index % 2 == 0 else "client"
        if turn.get("role") != expected_role:
            raise ValueError(f"utterance {index + 1} must have role {expected_role}")
        if not str(turn.get("utterance", "")).strip():
            raise ValueError(f"utterance {index + 1} has empty text")
    if output_language == "Korean":
        joined = "\n".join(str(turn.get("utterance", "")) for turn in dialogue[:4])
        if not has_hangul(joined):
            raise ValueError("Korean output must contain Hangul")


def make_news_id(stem: str, existing_ids: set[str]) -> str:
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


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")


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
    parser.add_argument("--raw-input", type=Path, default=DEFAULT_RAW_INPUT)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL)
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

    records, input_files = read_jsonl_dataset(args.raw_input)
    rows = flatten_dataset(records)
    examples = collect_dialogues(args.dialogue_dir, args.dialogue)
    client: OpenAI | None = None
    backend_used = ""
    output_records: list[dict[str, Any]] = []
    existing_ids = {str(record.get("news_id", "")) for record in records}

    for stem, dialogue_path in examples:
        print(f"Processing {stem}: dialogue={dialogue_path}", flush=True)
        dialogue = read_text(dialogue_path)
        output_language = desired_language(dialogue)
        item_generated_dir = args.generated_dir / stem

        if args.profile_json:
            profile = normalize_profile(json.loads(args.profile_json.read_text(encoding="utf-8")))
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
            raw_input=args.raw_input,
            cache_dir=args.cache_dir,
            top_k=args.top_k,
            backend=args.embedding_backend,
            sbert_model=args.sbert_model,
        )
        write_json(
            item_generated_dir / "similar_rows.json",
            {
                "source_dialogue": str(dialogue_path),
                "source_dataset_files": [str(path) for path in input_files],
                "backend": backend_used,
                "match_fields": PROFILE_FIELDS,
                "query_profile": {field: profile.get(field, "") for field in PROFILE_FIELDS},
                "rows": [compact_row(row) for row in similar_rows],
            },
        )
        if not args.retrieve_only:
            if client is None:
                client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            output_record = synthesize_record(
                client,
                model=args.dialogue_model,
                profile=profile,
                similar_rows=similar_rows,
                output_language=output_language,
            )
            output_record["news_id"] = make_news_id(stem, existing_ids)
            output_record = {
                "news_id": output_record["news_id"],
                "language": output_record["language"],
                "description": output_record["description"],
                "profile": output_record["profile"],
                "dialogue": output_record["dialogue"],
            }
            output_records.append(output_record)
            write_json(item_generated_dir / "sample.json", output_record)
        print(f"  Profile: {item_generated_dir / 'profile.json'}", flush=True)
        print(f"  Similar rows: {item_generated_dir / 'similar_rows.json'}", flush=True)
        if not args.retrieve_only:
            print(f"  Sample: {item_generated_dir / 'sample.json'}", flush=True)

    print(f"Processed examples: {len(examples)}")
    print(f"Embedding backend: {backend_used}")
    if not args.retrieve_only:
        write_jsonl(args.output_jsonl, output_records)
        print(f"Wrote records: {len(output_records)}")
        print(f"Output JSONL: {args.output_jsonl}")


if __name__ == "__main__":
    main()
