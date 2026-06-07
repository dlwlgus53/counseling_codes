#!/usr/bin/env python3
"""Generate additional MIRROR-style CSV rows from dialogue/image examples.

Pipeline:
1. Extract a MIRROR-style client profile from each dialogue + image with GPT-4o.
2. Retrieve the 3 most similar profiles from data/raw/train_ko.csv.
3. Ask GPT-4o to synthesize a new MIRROR-style row using the new profile and
   the retrieved dialogue examples.
4. Save all rows as data/raw/add_dataset.csv.

The retrieval cache uses SBERT when sentence-transformers is installed; otherwise
it falls back to a fast scikit-learn TF-IDF cache.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openai import OpenAI


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ADD_DIR = ROOT_DIR / "data/add_dataset"
DEFAULT_TRAIN_CSV = ROOT_DIR / "data/raw/train_ko.csv"
DEFAULT_OUTPUT_CSV = ROOT_DIR / "data/raw/add_dataset.csv"
DEFAULT_GENERATED_DIR = DEFAULT_ADD_DIR / "generated"
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp"]

FIELDNAMES = [
    "idx",
    "model",
    "personal_info",
    "personality",
    "distorted_thought",
    "thinking_trap",
    "reason_for_seeking_counseling",
    "cbt_plan",
    "dialogue",
    "proc_dialogue",
    "turn",
    "attitude",
    "name",
    "gender",
    "age",
    "age_group",
    "img_path",
    "identity",
    "celeb_male",
    "celeb_young",
    "region",
    "face_confidence",
    "dominant_gender",
    "emotion",
    "dominant_emotion",
    "n_image",
    "cbt_technique",
]

PROFILE_FIELDS = [
    "distorted_thought",
    "cbt_plan",
]

REQUIRED_PROFILE_FIELDS = [
    "personal_info",
    "personality",
    "distorted_thought",
    "thinking_trap",
    "reason_for_seeking_counseling",
    "cbt_plan",
]

PERSONALITY_TEXT = {
    "No Resistance": (
        "The client exhibits the personality trait of 'No Resistance'. "
        "Open to discussing feelings. Eager to explore new ideas. "
        "Willing to accept feedback. Shows commitment to the therapy process."
    ),
    "Emotional Resistance": (
        "The client exhibits the personality trait of 'Emotional Resistance'. "
        "Tends to avoid expressing emotions. Finds it difficult to show anxiety. "
        "Struggles to talk about past wounds. Acts indifferent towards emotions."
    ),
    "Cognitive Resistance": (
        "The client exhibits the personality trait of 'Cognitive Resistance'. "
        "Resists acknowledging their problems. Has a strong negative self-image. "
        "Easily rejects the therapist's advice. Maintains distorted thought patterns during therapy."
    ),
}


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def image_to_data_url(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix or "png"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{payload}"


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


def canonical_personality(value: str, fallback: str = "Emotional Resistance") -> str:
    for label, text in PERSONALITY_TEXT.items():
        if label.lower() in value.lower():
            return text
    return PERSONALITY_TEXT[fallback]


def extract_profile(
    client: OpenAI,
    *,
    model: str,
    dialogue: str,
    image_path: Path,
    output_language: str,
) -> dict[str, Any]:
    system = (
        "You create synthetic CBT counseling dataset metadata. "
        "Return only valid JSON. Do not diagnose; infer only dataset-style fields."
    )
    user_text = f"""
From the given counseling dialogue and client image, create a MIRROR-style profile.

Language rule:
- If the source dialogue is English-only, write all generated clinical fields in English.
- Otherwise, match train_ko.csv style: personal_info and personality may use
  the dataset's English sentence style, but distorted_thought, thinking_trap,
  reason_for_seeking_counseling, cbt_plan, attitude, dominant_emotion label
  interpretation, and cbt_technique must be Korean.

Return JSON with exactly these keys:
personal_info, personality, distorted_thought, thinking_trap,
reason_for_seeking_counseling, cbt_plan, attitude, name, gender, age,
age_group, dominant_gender, dominant_emotion, cbt_technique.

Match the style of the existing train_ko.csv rows:
- personal_info is a one-paragraph demographic sentence.
- personality uses one of: No Resistance, Emotional Resistance, Cognitive Resistance.
- distorted_thought is the client's main automatic thought.
- thinking_trap is a comma-separated list of CBT cognitive distortions.
- For Korean output, thinking_trap and cbt_technique must be Korean labels,
  e.g. 긍정적인 것 무시하기, 결론을 성급히 내리기: 마음 읽기, 재앙화, 인지 재구성.
- cbt_plan is a numbered CBT plan.
- gender is a string shaped like a Python dict with Woman/Man percentages when possible.
- age and age_group are strings.
- dominant_gender is Woman or Man when inferable.
- dominant_emotion is one of angry, disgust, fear, happy, sad, surprise, neutral when inferable.

Output language: {output_language}

Dialogue:
{dialogue}
"""
    base_messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
            ],
        },
    ]
    messages = base_messages
    last_profile: dict[str, Any] = {}
    for attempt in range(3):
        last_profile = openai_json(client, model=model, messages=messages, temperature=0.2)
        missing = missing_fields(last_profile, REQUIRED_PROFILE_FIELDS)
        if not missing:
            last_profile["personality"] = canonical_personality(str(last_profile.get("personality", "")))
            return last_profile
        messages = base_messages + [
            {
                "role": "assistant",
                "content": json.dumps(last_profile, ensure_ascii=False),
            },
            {
                "role": "user",
                "content": (
                    "The previous JSON was incomplete or empty. "
                    f"Missing fields: {missing}. Return a complete JSON object now."
                ),
            },
        ]
    raise ValueError(f"Profile extraction failed; missing fields: {missing_fields(last_profile, REQUIRED_PROFILE_FIELDS)}")


def row_profile_text(row: dict[str, Any]) -> str:
    parts = []
    for field in PROFILE_FIELDS:
        value = str(row.get(field, "")).strip()
        if value:
            parts.append(f"{field}: {value}")
    return "\n".join(parts)


def dataset_fingerprint(csv_path: Path) -> str:
    stat = csv_path.stat()
    raw = f"{csv_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def retrieve_similar_tfidf(
    train_df: pd.DataFrame,
    profile_text: str,
    *,
    train_csv: Path,
    cache_dir: Path,
    top_k: int,
) -> list[dict[str, Any]]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"profile_tfidf_{dataset_fingerprint(train_csv)}.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as f:
            payload = pickle.load(f)
        vectorizer = payload["vectorizer"]
        matrix = payload["matrix"]
        texts = payload["texts"]
    else:
        texts = [row_profile_text(row) for row in train_df.to_dict(orient="records")]
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2)
        matrix = vectorizer.fit_transform(texts)
        with cache_path.open("wb") as f:
            pickle.dump({"vectorizer": vectorizer, "matrix": matrix, "texts": texts}, f)

    query = vectorizer.transform([profile_text])
    scores = cosine_similarity(query, matrix)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [
        {
            "rank": rank + 1,
            "score": float(scores[index]),
            "row_index": int(index),
            **train_df.iloc[int(index)].to_dict(),
        }
        for rank, index in enumerate(top_indices)
    ]


def retrieve_similar_sbert(
    train_df: pd.DataFrame,
    profile_text: str,
    *,
    train_csv: Path,
    cache_dir: Path,
    top_k: int,
    model_name: str,
) -> list[dict[str, Any]]:
    from sentence_transformers import SentenceTransformer

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"profile_sbert_{dataset_fingerprint(train_csv)}_{model_name.replace('/', '_')}.pkl"
    model = SentenceTransformer(model_name)
    if cache_path.exists():
        with cache_path.open("rb") as f:
            embeddings = pickle.load(f)["embeddings"]
    else:
        texts = [row_profile_text(row) for row in train_df.to_dict(orient="records")]
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        with cache_path.open("wb") as f:
            pickle.dump({"embeddings": embeddings}, f)

    query = model.encode([profile_text], normalize_embeddings=True)[0]
    scores = np.asarray(embeddings) @ np.asarray(query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [
        {
            "rank": rank + 1,
            "score": float(scores[index]),
            "row_index": int(index),
            **train_df.iloc[int(index)].to_dict(),
        }
        for rank, index in enumerate(top_indices)
    ]


def retrieve_similar(
    train_df: pd.DataFrame,
    profile: dict[str, Any],
    *,
    train_csv: Path,
    cache_dir: Path,
    top_k: int,
    backend: str,
    sbert_model: str,
) -> tuple[str, list[dict[str, Any]]]:
    profile_text = row_profile_text(profile)
    if backend in {"auto", "sbert"}:
        try:
            return "sbert", retrieve_similar_sbert(
                train_df,
                profile_text,
                train_csv=train_csv,
                cache_dir=cache_dir,
                top_k=top_k,
                model_name=sbert_model,
            )
        except ModuleNotFoundError:
            if backend == "sbert":
                raise
    return "tfidf", retrieve_similar_tfidf(
        train_df,
        profile_text,
        train_csv=train_csv,
        cache_dir=cache_dir,
        top_k=top_k,
    )


def compact_example(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "idx": row.get("idx", ""),
        "profile": {field: row.get(field, "") for field in PROFILE_FIELDS},
        "dialogue": row.get("dialogue", ""),
        "proc_dialogue": row.get("proc_dialogue", ""),
        "turn": row.get("turn", ""),
        "name": row.get("name", ""),
        "age": row.get("age", ""),
        "dominant_gender": row.get("dominant_gender", ""),
        "dominant_emotion": row.get("dominant_emotion", ""),
    }


def synthesize_row(
    client: OpenAI,
    *,
    model: str,
    profile: dict[str, Any],
    similar_rows: list[dict[str, Any]],
    source_dialogue: str,
    image_path: Path,
    output_language: str,
) -> dict[str, Any]:
    system = (
        "You generate one synthetic MIRROR-style CBT counseling CSV row. "
        "Return only valid JSON."
    )
    examples = [compact_example(row) for row in similar_rows]
    user_text = f"""
Create one new row for train_ko.csv.

You must combine:
1. The newly extracted profile.
2. The style/structure of the 3 most similar existing examples.
3. The source dialogue's topic and counseling flow.

Hard requirements:
- Return JSON with exactly these columns:
{FIELDNAMES}
- personality must use exactly one of these existing dataset strings:
  {json.dumps(list(PERSONALITY_TEXT.values()), ensure_ascii=False)}
- The final dialogue should be a new dialogue, not a copy. Keep the same line style:
  Therapist: [stage direction] utterance
  Client: [stage direction] utterance
- Do not copy the source dialogue line-by-line. Preserve the theme and CBT flow,
  but change the wording, stage directions, ordering details, and therapeutic
  questions enough that it is clearly a newly written session.
- If output language is Korean, dialogue, distorted_thought, thinking_trap,
  reason_for_seeking_counseling, cbt_plan, and cbt_technique should be Korean.
- For Korean output, do not leave thinking_trap or cbt_technique in English.
- proc_dialogue should be a Python-list-like string of dictionaries with keys
  speaker, stage_direction, statement. It may be English or Korean, but should
  align turn-by-turn with dialogue.
- turn is the number of Therapist turns.
- model must be "{model}".
- idx should start with "add-" and be filesystem-safe.
- img_path must be "{image_path}".
- identity can be "add-example".
- region can be empty if not measured.
- face_confidence can be empty if not measured.
- n_image can be "1" for now.
- Do not mention that this was generated.

Output language: {output_language}

New profile:
{json.dumps(profile, ensure_ascii=False, indent=2)}

Similar existing examples:
{json.dumps(examples, ensure_ascii=False, indent=2)}

Source dialogue:
{source_dialogue}
"""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_text},
    ]
    row = openai_json(client, model=model, messages=messages, temperature=0.7)
    return normalize_output_row(row, image_path=image_path, model=model, profile=profile)


def normalize_output_row(
    row: dict[str, Any],
    *,
    image_path: Path,
    model: str,
    profile: dict[str, Any] | None = None,
) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for field in FIELDNAMES:
        value = row.get(field, "")
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        normalized[field] = str(value)

    normalized["model"] = model
    normalized["img_path"] = str(image_path)
    normalized["personality"] = canonical_personality(
        normalized.get("personality", ""),
        fallback="Emotional Resistance",
    )
    if profile:
        for field in [
            "personal_info",
            "distorted_thought",
            "thinking_trap",
            "reason_for_seeking_counseling",
            "cbt_plan",
            "attitude",
            "dominant_gender",
            "dominant_emotion",
            "cbt_technique",
        ]:
            if not normalized.get(field) and profile.get(field):
                normalized[field] = str(profile[field])
    if not normalized["idx"]:
        normalized["idx"] = "add-example"
    if not normalized["identity"]:
        normalized["identity"] = "add-example"
    if not normalized["n_image"]:
        normalized["n_image"] = "1"
    normalized["turn"] = str(sum(1 for line in normalized["dialogue"].splitlines() if line.startswith("Therapist:")))
    return normalized


def write_single_row_csv(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerow(row)


def write_rows_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def find_image_for_stem(image_dir: Path, stem: str) -> Path:
    for extension in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    expected = ", ".join(str(image_dir / f"{stem}{extension}") for extension in IMAGE_EXTENSIONS)
    raise FileNotFoundError(f"No matching image for dialogue stem '{stem}'. Expected one of: {expected}")


def collect_examples(args: argparse.Namespace) -> list[tuple[str, Path, Path]]:
    if args.dialogue:
        dialogue_path = args.dialogue
        image_path = args.image or find_image_for_stem(args.image_dir, dialogue_path.stem)
        return [(dialogue_path.stem, dialogue_path, image_path)]

    dialogue_paths = sorted(args.dialogue_dir.glob("*.txt"))
    if not dialogue_paths:
        raise FileNotFoundError(f"No dialogue .txt files found in {args.dialogue_dir}")
    return [
        (dialogue_path.stem, dialogue_path, find_image_for_stem(args.image_dir, dialogue_path.stem))
        for dialogue_path in dialogue_paths
    ]


def make_unique_ids(rows: list[dict[str, str]]) -> None:
    seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        base = row.get("idx", "").strip() or f"add-{index}"
        candidate = base
        suffix = 2
        while candidate in seen:
            candidate = f"{base}-{suffix}"
            suffix += 1
        row["idx"] = candidate
        seen.add(candidate)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dialogue-dir", type=Path, default=DEFAULT_ADD_DIR / "raw/dialogue")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_ADD_DIR / "raw/image")
    parser.add_argument("--dialogue", type=Path, default=None, help="Process one dialogue file instead of all dialogue/*.txt files.")
    parser.add_argument("--image", type=Path, default=None, help="Image for --dialogue. If omitted, matches by filename stem.")
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
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


def process_example(
    *,
    args: argparse.Namespace,
    client: OpenAI | None,
    train_df: pd.DataFrame,
    stem: str,
    dialogue_path: Path,
    image_path: Path,
) -> tuple[OpenAI | None, str, dict[str, str] | None]:
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
            image_path=image_path,
            output_language=output_language,
        )
    write_json(item_generated_dir / "profile.json", profile)

    backend_used, similar_rows = retrieve_similar(
        train_df,
        profile,
        train_csv=args.train_csv,
        cache_dir=args.cache_dir,
        top_k=args.top_k,
        backend=args.embedding_backend,
        sbert_model=args.sbert_model,
    )
    write_json(
        item_generated_dir / "similar_rows.json",
        {
            "source_dialogue": str(dialogue_path),
            "source_image": str(image_path),
            "backend": backend_used,
            "rows": [
                {
                    "rank": row["rank"],
                    "score": row["score"],
                    "row_index": row["row_index"],
                    "idx": row.get("idx", ""),
                    "profile": {field: row.get(field, "") for field in PROFILE_FIELDS},
                }
                for row in similar_rows
            ],
        },
    )
    if args.retrieve_only:
        return client, backend_used, None

    if client is None:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    output_row = synthesize_row(
        client,
        model=args.dialogue_model,
        profile=profile,
        similar_rows=similar_rows,
        source_dialogue=dialogue,
        image_path=image_path.resolve(),
        output_language=output_language,
    )
    return client, backend_used, output_row


def main() -> None:
    args = parse_args()
    if args.profile_json and not args.dialogue:
        raise ValueError("--profile-json can only be used with --dialogue")

    examples = collect_examples(args)
    train_df = pd.read_csv(args.train_csv, dtype=str).fillna("")
    client: OpenAI | None = None
    output_rows: list[dict[str, str]] = []
    backend_used = ""

    for stem, dialogue_path, image_path in examples:
        print(f"Processing {stem}: dialogue={dialogue_path} image={image_path}", flush=True)
        client, backend_used, output_row = process_example(
            args=args,
            client=client,
            train_df=train_df,
            stem=stem,
            dialogue_path=dialogue_path,
            image_path=image_path,
        )
        if output_row is not None:
            output_rows.append(output_row)
        print(f"  Profile: {args.generated_dir / stem / 'profile.json'}", flush=True)
        print(f"  Similar rows: {args.generated_dir / stem / 'similar_rows.json'}", flush=True)

    if args.retrieve_only:
        print(f"Processed examples: {len(examples)}")
        print(f"Embedding backend: {backend_used}")
        return

    make_unique_ids(output_rows)
    write_rows_csv(args.output_csv, output_rows)
    print(f"Processed examples: {len(examples)}")
    print(f"Wrote rows: {len(output_rows)}")
    print(f"Embedding backend: {backend_used}")
    print(f"Output CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
