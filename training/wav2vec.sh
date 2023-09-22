# ~/.local/bin/deepspeed wav2vec2-finetune.py \
# --deepspeed ds_config_zero3.json \
# --model_name_or_path="facebook/wav2vec2-large-xlsr-53" \
# --output_dir="wav2vec2" \
# --num_train_epochs="1" \
# --per_device_train_batch_size="16" \
# --learning_rate="2e-5" \
# --evaluation_strategy="steps" \
# --save_steps="400" \
# --eval_steps="400" \
# --logging_steps="50" \
# --layerdrop="0.0" \
# --save_total_limit="2" \
# --gradient_checkpointing \
# --do_train

python3 wav2vec2-finetune.py \
--model_name_or_path="facebook/wav2vec2-large-xlsr-53" \
--output_dir="./wav2vec2-bengali-xlsr-val+train" \
--num_train_epochs="20" \
--per_device_train_batch_size="5" \
--learning_rate="2e-5" \
--evaluation_strategy="steps" \
--save_steps="400" \
--eval_steps="2000" \
--logging_steps="50" \
--layerdrop="0.0" \
--save_total_limit="2" \
--gradient_checkpointing \
--do_train

# python3 wav2vec2-finetune.py \
# --model_name_or_path="facebook/wav2vec2-large-xlsr-53" \
# --output_dir="./wav2vec2-bengali-xlsr-mask" \
# --num_train_epochs="20" \
# --per_device_train_batch_size="16" \
# --learning_rate="2e-5" \
# --evaluation_strategy="steps" \
# --save_steps="400" \
# --eval_steps="2000" \
# --logging_steps="50" \
# --layerdrop="0.0" \
# --save_total_limit="2" \
# --gradient_checkpointing \
# --mask_time_prob=0.2 \
# --mask_feature_prob=0.2 \
# --do_train \
# --bf16

# python3 wav2vec2-finetune.py \
# --model_name_or_path="facebook/wav2vec2-large-xlsr-53" \
# --output_dir="./wav2vec2-bengali-xlsr-id-4" \
# --num_train_epochs="20" \
# --per_device_train_batch_size="5" \
# --learning_rate="2e-5" \
# --evaluation_strategy="steps" \
# --save_steps="2000" \
# --eval_steps="3000" \
# --logging_steps="1000" \
# --layerdrop="0.0" \
# --save_total_limit="2" \
# --gradient_checkpointing \
# --mask_time_prob="0.75" \
# --mask_time_length="10" \
# --mask_feature_prob="0.25" \
# --mask_feature_length="64" \
# --do_train \
# --bf16
