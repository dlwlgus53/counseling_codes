# MIRROR Train CSV Korean Translation with OpenAI Batch API

이 문서는 `/home/jihyunlee/mirror/MIRROR_code/data/processed/train.csv`를 OpenAI 공식 Batch API로 한국어 번역하는 실행 흐름을 정리한 것입니다.

## 0. Conda 환경 활성화

이 스크립트는 `collabllm` conda 환경의 OpenAI 공식 SDK를 사용합니다.

```bash
conda activate collabllm
```

## 1. 컬럼 확인

```bash
python /home/jihyunlee/mirror/MIRROR_code/mirror/translate.py inspect
```

현재 CSV row 수는 `3072`개입니다.

## 2. 10개만 테스트 Batch 준비

비용이 나가지 않는 단계입니다. OpenAI에 제출할 JSONL 파일만 만듭니다.

```bash
python /home/jihyunlee/mirror/MIRROR_code/mirror/translate.py prepare --num-rows 10
```

생성 위치:

```text
/home/jihyunlee/mirror/translate/train_ko_batch/batch_input_000.jsonl
```

## 3. 10개 Batch 제출

이 단계부터 OpenAI API 비용이 발생할 수 있습니다.

```bash
python /home/jihyunlee/mirror/MIRROR_code/mirror/translate.py submit
```

제출 정보는 여기에 저장됩니다.

```text
/home/jihyunlee/mirror/translate/train_ko_batch/batch_jobs.json
```

## 4. Batch 상태 확인

```bash
python /home/jihyunlee/mirror/MIRROR_code/mirror/translate.py status
```

`status=completed`가 나오면 다운로드할 수 있습니다.

## 5. 결과 다운로드

```bash
python /home/jihyunlee/mirror/MIRROR_code/mirror/translate.py download
```

다운로드 위치:

```text
/home/jihyunlee/mirror/translate/train_ko_batch/outputs/
```

## 6. 번역 결과를 CSV에 병합

```bash
python /home/jihyunlee/mirror/MIRROR_code/mirror/translate.py apply
```

기본 출력 파일:

```text
/home/jihyunlee/mirror/MIRROR_code/data/processed/train_ko.csv
```

기본 동작은 번역 대상 원본 컬럼을 한국어 값으로 교체합니다.
즉 `dialogue` 컬럼에는 영어 원문 대신 한국어 번역문이 들어갑니다.

원본 영어 컬럼도 같이 보존하고 싶으면 `--keep-original`을 붙입니다.
이 경우 번역값은 `*_ko` 컬럼으로 추가됩니다.

예:

```text
dialogue -> dialogue_ko
cbt_plan -> cbt_plan_ko
```

## 전체 데이터 실행

테스트 10개가 잘 되면 전체 3072개를 준비합니다.

```bash
python /home/jihyunlee/mirror/MIRROR_code/mirror/translate.py prepare --num-rows -1
python /home/jihyunlee/mirror/MIRROR_code/mirror/translate.py submit
python /home/jihyunlee/mirror/MIRROR_code/mirror/translate.py status
python /home/jihyunlee/mirror/MIRROR_code/mirror/translate.py download
python /home/jihyunlee/mirror/MIRROR_code/mirror/translate.py apply
```

10개 테스트 결과 파일을 10개 row만 만들고 싶으면:

```bash
python /home/jihyunlee/mirror/MIRROR_code/mirror/translate.py apply --only-translated-rows --output /home/jihyunlee/mirror/MIRROR_code/data/processed/train_ko_test10.csv
```

## API Key 선택

기본값은 `/home/jihyunlee/mirror/configs/api.json`의 `api-key`입니다.

다른 key field를 쓰고 싶으면 모든 명령에서 전역 옵션을 command 앞에 붙입니다.

```bash
python /home/jihyunlee/mirror/MIRROR_code/mirror/translate.py --api-key-field lab-api-key1 submit
```

주의: `--api-key-field` 같은 전역 옵션은 `submit`, `prepare` 같은 command 앞에 위치해야 합니다.

## 기본 번역 대상 컬럼

```text
distorted_thought
thinking_trap
reason_for_seeking_counseling
cbt_plan
dialogue
attitude
cbt_technique
```

## 코드 흐름

```text
train.csv
  |
  | prepare --num-rows 10 또는 --num-rows -1
  v
batch_input_000.jsonl
  |
  | submit
  | - client.files.create(..., purpose="batch")
  | - client.batches.create(..., endpoint="/v1/chat/completions")
  v
batch_jobs.json
  |
  | status
  | - client.batches.retrieve(batch_id)
  v
completed batch
  |
  | download
  | - client.files.content(output_file_id)
  v
outputs/*_output.jsonl
  |
  | apply
  | - custom_id(row-N) 기준으로 원본 row와 매칭
  | - 기본값: 번역 대상 원본 컬럼을 한국어 값으로 교체
  | - --keep-original 사용 시: 번역 결과를 *_ko 컬럼으로 추가
  v
train_ko.csv
```

## 주요 함수 역할

`read_api_key`
: `configs/api.json`에서 OpenAI API key를 읽습니다.

`make_client`
: `collabllm` 환경에 설치된 OpenAI 공식 SDK의 `OpenAI` client를 생성합니다.

`build_request`
: CSV 한 row를 OpenAI Batch API용 JSONL 한 줄로 변환합니다.

`write_jsonl_shards`
: `--num-rows` 값에 따라 10개 또는 전체 row를 JSONL batch input으로 만듭니다.

`submit_batches`
: JSONL 파일을 OpenAI Files API에 업로드하고 Batch job을 생성합니다.

`show_status`
: Batch job의 진행 상태와 성공/실패 request 수를 확인합니다.

`download_outputs`
: 완료된 Batch output JSONL을 다운로드합니다.

`merge_outputs`
: 다운로드한 번역 결과를 원본 CSV에 병합합니다. 기본값은 번역 대상 컬럼을 한국어 값으로 교체하는 방식입니다.
