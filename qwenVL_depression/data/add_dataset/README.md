# Depression add_dataset

이 폴더는 우울/CBT MIRROR 데이터에 새 상담 예시를 추가하기 위한 작업 공간입니다. Depression은 세 프로젝트 중 유일하게 이미지가 함께 들어가는 멀티모달 구조입니다.

## 입력 구조

```text
raw/dialogue/{stem}.txt
raw/image/{stem}.png
```

`dialogue`와 `image` 파일은 확장자를 제외한 이름이 같아야 합니다. 예를 들어 `raw/dialogue/example.txt`를 넣었다면 `raw/image/example.png`가 필요합니다.

## 실행

```bash
cd /home/jihyunlee/mirror/MIRROR_code/qwenVL_depression
bash scripts/generate_new_dataset.sh
```

하나의 dialogue/image만 처리하려면:

```bash
python src/build_add_dataset_example.py \
  --dialogue data/add_dataset/raw/dialogue/example.txt \
  --image data/add_dataset/raw/image/example.png
```

## 코드 흐름

1. `scripts/generate_new_dataset.sh`가 `mirror` conda 환경을 활성화하고 OpenAI API 키를 읽습니다.
2. `src/build_add_dataset_example.py`가 `raw/dialogue`와 `raw/image` 입력을 읽습니다.
3. GPT가 새 상담 예시의 profile을 추출합니다.
4. 추출한 profile의 `distorted_thought`, `cbt_plan` 등을 기준으로 `data/raw/train_ko.csv`에서 유사 예시를 검색합니다.
5. GPT가 새 profile, 원본 dialogue/image, 유사 예시를 바탕으로 MIRROR 형식의 CSV row를 생성합니다.
6. 생성된 row는 `data/raw/add_dataset.csv`에 저장됩니다.
7. 기본 설정에서는 `image_synthesis/run_image_synthesis.py`가 이미지 프롬프트와 합성 이미지 생성 과정을 이어서 실행합니다.
8. 이후 `scripts/finetune_mirror_kor_lora.sh`를 실행하면 `data/raw/train_ko.csv`와 `data/raw/add_dataset.csv`를 사용해 `data/mirror_ko_train.jsonl`을 만들고 LoRA 학습을 시작합니다.

## 생성 결과

```text
generated/{stem}/profile.json
generated/{stem}/similar_rows.json
data/raw/add_dataset.csv
```

이미지 합성까지 실행하면 아래 폴더도 사용됩니다.

```text
image_synthesis/annot_data/
image_synthesis/photomaker_prompts/
```

## 옵션

- `SKIP_IMAGE_SYNTHESIS=1`: 이미지 합성 단계를 건너뜁니다.
- `SKIP_GPT_IMAGE_ANNOTATION=1`: 이미지 annotation GPT 호출을 건너뜁니다.
- `SKIP_PHOTOMAKER=1`: PhotoMaker 이미지 생성을 건너뜁니다.
- `BUILD_QWEN_JSONL=1`: 추가 CSV만 별도 Qwen JSONL로 변환합니다.

