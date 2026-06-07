# MIRROR Code 

이 폴더는 MIRROR 계열 상담 데이터셋과 Qwen3-VL LoRA 학습 코드를 정리한 작업 디렉터리입니다. 현재 핵심 대상은 `depression`, `panic`, `PTSD` 세 가지 상담 도메인입니다.

## 주요 구성

- `qwenVL_depression/`: 우울/CBT MIRROR 한국어 데이터 기반 Qwen3-VL LoRA 학습 코드
- `qwenVL_panic/`: 공황/급성불안 응급상담 데이터 기반 Qwen3-VL LoRA 학습 코드
- `qwenVL_PTSD/`: PTSD/트라우마 기반 CBT 상담 데이터 기반 Qwen3-VL LoRA 학습 코드
- `configs/`: 로컬 설정 파일 위치

## API 키 설정

실제 API 키 파일은 인수인계 자료에서 제거했습니다.

OpenAI API를 사용하는 추가 데이터 생성 스크립트를 실행하려면 아래 템플릿을 복사해서 로컬에서만 채워 넣으세요.

```bash
cd /home/jihyunlee/mirror/MIRROR_code
cp configs/api.example.json configs/api.json
```

그다음 `configs/api.json`에 실제 키를 입력합니다.

```json
{
  "api-key": "YOUR_OPENAI_API_KEY"
}
```

## 공통 실행 환경

대부분의 Qwen3-VL 학습/생성 스크립트는 `mirror` conda 환경을 우선 사용합니다.

기존 서버에서는 아래처럼 활성화하면 됩니다.

```bash
conda activate mirror
```

새 서버에서 환경을 다시 만들 때는 다음 순서로 설치하세요.

```bash
conda create -n mirror python=3.10 -y
conda activate mirror
python -m pip install --upgrade pip
```

먼저 PyTorch를 서버의 CUDA/드라이버에 맞게 설치합니다. 기존 환경은 `torch 2.11.0+cu130` 계열을 사용했습니다. 새 서버 CUDA 버전에 맞는 설치 명령은 PyTorch 공식 설치 안내에 맞춰 조정하세요.

예시:

```bash
# 예시입니다. 실제 CUDA 버전에 맞게 설치 명령을 바꾸세요.
pip install torch torchvision torchaudio
```

그다음 이 저장소의 requirement를 설치합니다.

```bash
cd /home/jihyunlee/mirror/MIRROR_code
pip install -r requirements.txt
```

이미지 합성 보조 기능까지 사용할 경우 선택 의존성을 추가로 설치합니다.

```bash
pip install -r requirements_optional.txt
```

설치 확인:

```bash
python - <<'PY'
import torch
import transformers
import peft
import openai
import sentence_transformers

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("transformers:", transformers.__version__)
print("peft:", peft.__version__)
print("openai:", openai.__version__)
print("sentence-transformers:", sentence_transformers.__version__)
PY
```

추가 데이터 생성 스크립트는 OpenAI API를 사용합니다. 학습 스크립트는 API 키 없이 실행 가능하지만, Hugging Face 모델 다운로드와 GPU 환경이 필요합니다.


## Depression

위치:

```bash
cd /home/jihyunlee/mirror/MIRROR_code/qwenVL_depression
```

목적:

- MIRROR 한국어 우울/CBT 상담 데이터를 Qwen3-VL SFT 형식으로 변환
- 이미지가 포함된 멀티모달 상담 예시로 LoRA 학습
- 추가 dialogue/image 예시를 바탕으로 MIRROR 스타일의 추가 CSV row 생성

핵심 입력:

```text
data/raw/train_ko.csv
data/mirror_images/
```

이미지 파일은 GitHub에 포함하지 않습니다. 인수인계받은 이미지 다운로드 링크에서 받은 뒤, 압축을 풀어 아래 구조가 되도록 배치합니다.

```text
qwenVL_depression/data/mirror_images/{case_id}/turn:*.png
```

이미지 다운로드 링크:

```text
https://drive.google.com/file/d/1N5BHb7H3IuaJuRC2o-FXnL_z8PACCjyV/view?usp=sharing
```

학습용 변환 결과:

```text
data/mirror_ko_train.jsonl
```

### 새로운 데이터 추가시

미리 있어야 하는 것:

- OpenAI API 키가 설정된 `configs/api.json` 또는 `OPENAI_API_KEY` 환경변수
- 기준 데이터 `data/raw/train_ko.csv`
- 기준 이미지 폴더 `data/mirror_images/`
- 새 상담 대화 txt 파일: `data/add_dataset/raw/dialogue/{stem}.txt`
- 새 대화에 대응되는 이미지 파일: `data/add_dataset/raw/image/{stem}.png` 또는 `.jpg`, `.jpeg`, `.webp`

주의: dialogue 파일명과 image 파일명은 확장자를 제외한 stem이 같아야 합니다. 예를 들어 `raw/dialogue/example.txt`를 추가했다면 `raw/image/example.png` 같은 이미지가 있어야 합니다.

실행:

```bash
cd /home/jihyunlee/mirror/MIRROR_code/qwenVL_depression
bash data/add_dataset/generate_new_dataset.sh
```

하나의 파일만 처리하려면:

```bash
python src/build_add_dataset_example.py \
  --dialogue data/add_dataset/raw/dialogue/example.txt \
  --image data/add_dataset/raw/image/example.png
```

생성 과정:

1. 입력 dialogue와 image를 GPT에 넣어 MIRROR 스타일 profile을 추출합니다.
2. 추출한 profile 중 `distorted_thought`, `cbt_plan`을 중심으로 `data/raw/train_ko.csv`에서 유사 예시 3개를 검색합니다.
3. 새 profile, 원본 dialogue/image, 유사 예시를 바탕으로 GPT가 MIRROR CSV row를 합성합니다.
4. 생성된 row들을 하나의 추가 CSV로 저장합니다.
5. 이후 `scripts/finetune_mirror_kor_lora.sh`를 실행하면 학습용 `data/mirror_ko_train.jsonl`을 다시 만들고 LoRA 학습을 시작합니다.

생성 결과:

```text
data/raw/add_dataset.csv
data/add_dataset/generated/{stem}/profile.json
data/add_dataset/generated/{stem}/similar_rows.json
```

이미지 합성 관련 작업물:

```text
data/add_dataset/image_synthesis/
```

학습 실행:

```bash
bash scripts/finetune_mirror_kor_lora.sh
```

간단한 1-step smoke test:

```bash
WANDB_MODE=offline MAX_STEPS=1 SAVE_STEPS=1 LOGGING_STEPS=1 IO_LOG_STEPS=1 bash scripts/run.sh
```

체크포인트:

```text
checkpoints/qwen3-vl-8b-mirror-kor-lora/
```

주의:

- Depression은 세 프로젝트 중 유일하게 이미지 경로와 이미지 파일을 학습 입력에 사용합니다.
- GitHub에는 이미지 파일을 올리지 않으므로, 학습 전 `data/mirror_images/`를 별도로 내려받아 복원해야 합니다.

## Panic

위치:

```bash
cd /home/jihyunlee/mirror/MIRROR_code/qwenVL_panic
```

목적:

- 공황 발작, 급성 불안, 신체 증상으로 위기감을 느끼는 사용자를 돕는 한국어 응급상담 모델 학습
- 텍스트-only Qwen3-VL LoRA 학습
- 새로운 panic dialogue txt에서 profile을 추출하고 유사 예시를 검색해 추가 raw JSON 생성

핵심 입력:

```text
data/raw/panic_dataset.json
```

기본 학습 빌더는 `data/raw/*.json`을 모두 읽습니다. 따라서 새로 생성된 `data/raw/add_dataset.json`이 있으면 기존 `panic_dataset.json`과 함께 병합됩니다.

학습용 변환 결과:

```text
data/panic_ko_train.jsonl
```

### 새로운 데이터 추가시

미리 있어야 하는 것:

- OpenAI API 키가 설정된 `configs/api.json` 또는 `OPENAI_API_KEY` 환경변수
- 기준 raw 데이터 `data/raw/panic_dataset.json`
- 새 상담 대화 txt 파일: `data/add_dataset/raw/dialogue/{stem}.txt`
- 유사도 검색용 패키지: 기본은 `sentence-transformers`, 없거나 가볍게 돌릴 때는 `--embedding-backend tfidf` 사용 가능

실행:

```bash
cd /home/jihyunlee/mirror/MIRROR_code/qwenVL_panic
bash data/add_dataset/generate_new_dataset.sh
```

하나의 파일만 처리하려면:

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

생성 과정:

1. 입력 dialogue에서 panic-style client profile을 추출합니다.
2. profile의 `trigger`, `physical_symptom`, `catastrophic_thought`, `emotional_react`를 기준으로 기존 `panic_dataset.json`에서 유사 예시 3개를 검색합니다.
3. 새 profile과 유사 예시의 상담 흐름을 바탕으로 GPT가 새로운 panic 상담 plan과 dialogue를 생성합니다.
4. 생성 결과를 panic raw JSON 구조로 정규화해 저장합니다.
5. 이후 `scripts/finetune_panic_kor_lora.sh`를 실행하면 `data/raw/*.json`을 모두 병합해 `data/panic_ko_train.jsonl`을 만들고 LoRA 학습을 시작합니다.

생성 결과:

```text
data/raw/add_dataset.json
data/add_dataset/generated/{stem}/profile.json
data/add_dataset/generated/{stem}/similar_rows.json
data/add_dataset/generated/{stem}/sample.json
```

학습 실행:

```bash
bash scripts/finetune_panic_kor_lora.sh
```

간단한 1-step smoke test:

```bash
WANDB_MODE=offline MAX_STEPS=1 SAVE_STEPS=1 LOGGING_STEPS=1 IO_LOG_STEPS=1 bash scripts/run.sh
```

체크포인트:

```text
checkpoints/qwen3-vl-8b-panic-kor-lora/
```

상담 흐름:

- 공감과 정서 확인
- 현재 위치와 안전 확인
- 사람이 많거나 시끄러운 곳이면 더 조용하고 안전한 곳으로 이동 유도
- 호흡/그라운딩으로 안정화
- 죽을 것 같다는 공포가 있을 때 신체 경보 반응으로 설명하되 진단은 단정하지 않음
- 위험 신호가 있으면 119, 응급실, 주변 도움, 전문 진료 안내

## PTSD

위치:

```bash
cd /home/jihyunlee/mirror/MIRROR_code/qwenVL_PTSD
```

목적:

- PTSD/트라우마 기반 CBT 상담 데이터를 Qwen3-VL SFT 형식으로 변환
- 텍스트-only Qwen3-VL LoRA 학습
- 새로운 PTSD dialogue txt에서 trauma profile을 추출하고 유사 예시를 검색해 추가 raw JSONL 생성

핵심 입력:

```text
data/raw/PTSD.jsonl
```

현재 `PTSD.jsonl`에는 `news_id`, `language`, `description`, `profile`, `dialogue`가 함께 저장되어 있습니다.

`profile`에는 다음 정보가 들어 있습니다.

```text
name
gender
age
trauma_event
daily_difficulty_en / daily_difficulty_ko
belief_state_en / belief_state_ko
counseling_goal_en / counseling_goal_ko
```

기본 학습 빌더는 `data/raw/*.jsonl`을 모두 읽습니다. 따라서 새로 생성된 `data/raw/add_dataset.jsonl`이 있으면 기존 `PTSD.jsonl`과 함께 병합됩니다.

학습용 변환 결과:

```text
data/ptsd_ko_train.jsonl
```

### 새로운 데이터 추가시

미리 있어야 하는 것:

- OpenAI API 키가 설정된 `configs/api.json` 또는 `OPENAI_API_KEY` 환경변수
- 기준 raw 데이터 `data/raw/PTSD.jsonl`
- `PTSD.jsonl` 안의 각 record에 `profile` 정보가 포함되어 있어야 함
- 새 상담 대화 txt 파일: `data/add_dataset/raw/dialogue/{stem}.txt`
- 유사도 검색용 패키지: 기본은 `sentence-transformers`, 없거나 가볍게 돌릴 때는 `--embedding-backend tfidf` 사용 가능

실행:

```bash
cd /home/jihyunlee/mirror/MIRROR_code/qwenVL_PTSD
bash data/add_dataset/generate_new_dataset.sh
```

하나의 파일만 처리하려면:

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

생성 과정:

1. 입력 dialogue에서 PTSD-style trauma profile을 추출합니다.
2. 추출 profile에는 `trauma_event`, `daily_difficulty`, `belief_state`, `counseling_goal`이 포함됩니다.
3. `trauma_event.event_description`, `daily_difficulty_ko`, `belief_state_ko`, `counseling_goal_ko`를 기준으로 기존 `PTSD.jsonl`에서 유사 예시 3개를 검색합니다.
4. 새 profile과 유사 예시의 상담 흐름을 바탕으로 GPT가 새로운 PTSD 상담 dialogue를 생성합니다.
5. 생성 결과를 PTSD raw JSONL 구조로 정규화해 저장합니다.
6. 이후 `scripts/finetune_ptsd_kor_lora.sh`를 실행하면 `data/raw/*.jsonl`을 모두 병합해 `data/ptsd_ko_train.jsonl`을 만들고 LoRA 학습을 시작합니다.

생성 결과:

```text
data/raw/add_dataset.jsonl
data/add_dataset/generated/{stem}/profile.json
data/add_dataset/generated/{stem}/similar_rows.json
data/add_dataset/generated/{stem}/sample.json
```

학습 데이터 JSONL 빌드:

```bash
python src/build_ptsd_ko_qwen_json.py
```

학습 실행:

```bash
bash scripts/finetune_ptsd_kor_lora.sh
```

간단한 1-step smoke test:

```bash
WANDB_MODE=offline MAX_STEPS=1 SAVE_STEPS=1 LOGGING_STEPS=1 IO_LOG_STEPS=1 bash scripts/run.sh
```

체크포인트:

```text
checkpoints/qwen3-vl-8b-ptsd-kor-lora/
```

상담 흐름:

- 일상에서 힘든 순간을 부드럽게 묻고 시작
- 트라우마를 처음부터 직접 캐묻지 않음
- trigger, avoidance behavior, maladaptive thought를 점진적으로 탐색
- broader belief state를 확인
- 공감과 검증 후 CBT식 질문으로 균형 잡힌 adaptive thought를 형성
- 작고 안전한 practice task로 마무리

## 체크포인트와 캐시

요청에 따라 체크포인트와 캐시 파일은 유지했습니다.

주요 체크포인트 위치:

```text
qwenVL_depression/checkpoints/
qwenVL_panic/checkpoints/
qwenVL_PTSD/checkpoints/
```

추가 데이터 검색 캐시 위치:

```text
qwenVL_depression/data/add_dataset/cache/
qwenVL_panic/data/add_dataset/cache/
qwenVL_PTSD/data/add_dataset/cache/
```

## 전달 전 주의사항

- `configs/api.json`은 실제 키가 들어가는 로컬 파일이므로 공유하지 마세요.
- 현재 인수인계 폴더에는 체크포인트와 캐시가 유지되어 있어 용량이 큽니다.
- `wandb/`, `__pycache__/`, `.pyc`, `.bak` 파일은 실행에 필수는 아니지만 현재 작업 상태 보존을 위해 그대로 두었습니다.
- 모델 가중치/adapter 파일은 용량이 크므로 전달 방식에 따라 별도 압축 또는 스토리지 공유가 필요할 수 있습니다.

## 빠른 확인 명령

Depression 학습 데이터 확인:

```bash
cd /home/jihyunlee/mirror/MIRROR_code/qwenVL_depression
python src/build_mirror_ko_qwen_json.py --csv data/raw/train_ko.csv --output /tmp/depression_smoke.jsonl
wc -l /tmp/depression_smoke.jsonl
```

Panic 학습 데이터 확인:

```bash
cd /home/jihyunlee/mirror/MIRROR_code/qwenVL_panic
python src/build_panic_ko_qwen_json.py --input data/raw --output /tmp/panic_smoke.jsonl
wc -l /tmp/panic_smoke.jsonl
```

PTSD 학습 데이터 확인:

```bash
cd /home/jihyunlee/mirror/MIRROR_code/qwenVL_PTSD
python src/build_ptsd_ko_qwen_json.py --input data/raw --output /tmp/ptsd_smoke.jsonl
wc -l /tmp/ptsd_smoke.jsonl
```
