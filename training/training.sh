# ~/.local/bin/deepspeed run_speech_recognition_ctc.py \
# --deepspeed ds_config_zero3.json \
# --model_name_or_path="./mms-1b-all" \
# --output_dir="./mms-1b" \
# --num_train_epochs="1" \
# --per_device_train_batch_size="50" \
# --learning_rate="2e-5" \
# --evaluation_strategy="steps" \
# --save_steps="400" \
# --eval_steps="400" \
# --logging_steps="50" \
# --layerdrop="0.0" \
# --save_total_limit="2" \
# --gradient_checkpointing \
# --do_train \
# --bf16

python3 run_speech_recognition_ctc.py \
--model_name_or_path="./mms-1b-all" \
--output_dir="./mms-1b" \
--num_train_epochs="1" \
--per_device_train_batch_size="16" \
--learning_rate="2e-5" \
--evaluation_strategy="steps" \
--save_steps="400" \
--eval_steps="400" \
--logging_steps="50" \
--layerdrop="0.0" \
--save_total_limit="2" \
--gradient_checkpointing \
--do_train \
--bf16