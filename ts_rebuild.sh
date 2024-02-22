#! /bin/bash
torchserve --stop

torch-model-archiver --model-name gte \
--version 1.0 \
--model-file my_model/model.safetensors \
--handler handler.py \
--extra-files "my_model/config.json,my_tokenizer/special_tokens_map.json,my_tokenizer/tokenizer_config.json,my_tokenizer/tokenizer.json,my_tokenizer/vocab.txt" \
--export-path model_store -f

torchserve --start --model-store model_store --models gte=gte.mar --ncs

#torchserve --stop
