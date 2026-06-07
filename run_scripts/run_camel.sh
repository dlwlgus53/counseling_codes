#!/bin/bash
export OPENAI_API_KEY=""

cd ..

CUDA_VISIBLE_DEVICES=0 python -m llm_therapist.run \
    --client_model_name gpt-3.5-turbo \
    --counselor_model_path /home/kimsubin/model/camel-llama3 \
    --input_data ../data/processed/test.csv