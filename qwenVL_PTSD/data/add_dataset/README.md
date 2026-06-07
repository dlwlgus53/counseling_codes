# PTSD add_dataset

이 폴더는 PTSD/트라우마 기반 CBT 상담 데이터에 새 텍스트 상담 예시를 추가하기 위한 작업 공간입니다. PTSD는 텍스트-only 구조라 이미지 입력이 없습니다.

## 입력 구조

```text
raw/dialogue/{stem}.txt
```

새로 만들 상담 대화를 txt 파일로 넣습니다. 파일명 stem은 생성 결과 폴더명과 sample id에 사용됩니다.

## 실행

```bash
cd /home/jihyunlee/mirror/MIRROR_code/qwenVL_PTSD
bash scripts/generate_new_dataset.sh
```

하나의 dialogue만 처리하려면:

```bash
python src/build_add_dataset_example.py \
  --dialogue data/add_dataset/raw/dialogue/example.txt
```

검색까지만 확인하려면:

```bash
python src/build_add_dataset_example.py \
  --dialogue data/add_dataset/raw/dialogue/example.txt \
  --retrieve-only
```

## 코드 흐름

1. `scripts/generate_new_dataset.sh`가 `mirror` conda 환경을 활성화하고 OpenAI API 키를 읽습니다.
2. `src/build_add_dataset_example.py`가 `raw/dialogue`의 txt 파일을 읽습니다.
3. GPT가 PTSD 상담 데이터 형식의 profile을 추출합니다.
4. profile의 trauma event, daily difficulty, belief state, counseling goal 정보를 기준으로 `data/raw/PTSD.jsonl`에서 유사 예시를 검색합니다.
5. GPT가 새 profile과 유사 예시의 상담 흐름을 바탕으로 새로운 PTSD 상담 dialogue를 생성합니다.
6. 생성 결과를 PTSD raw JSONL 구조로 정규화해 `data/raw/add_dataset.jsonl`에 저장합니다.
7. 이후 `scripts/finetune_ptsd_kor_lora.sh`를 실행하면 `data/raw/*.jsonl`을 모두 병합해 `data/ptsd_ko_train.jsonl`을 만들고 LoRA 학습을 시작합니다.

## 생성 결과

```text
generated/{stem}/profile.json
generated/{stem}/similar_rows.json
generated/{stem}/sample.json
data/raw/add_dataset.jsonl
```

## 유사도 검색 캐시

```text
cache/*.pkl
```

기본은 SBERT 검색을 사용합니다. 가볍게 확인할 때는 아래처럼 TF-IDF 검색으로 바꿀 수 있습니다.

```bash
python src/build_add_dataset_example.py \
  --dialogue data/add_dataset/raw/dialogue/example.txt \
  --embedding-backend tfidf
```

