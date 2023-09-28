CUDA_VISIBLE_DEVICES='1' WANDB_DISABLED=true \
python3 trainer.py \
--model conformer_rnnt_base \
--train_dataset /home/husein/speech-bahasa/malay-asr-train-shuffled.json \
--val_dataset /home/husein/ssd1/speech-bahasa/malay-asr-test.json \
--output_dir conformer_rnnt_base \
--learning_rate="2e-5" \
--save_total_limit 3 \
--save_steps 10000 \
--eval_steps 300 \
--logging_steps 300 \
--warmup_steps 100 \
--num_train_epochs 2 \
--layerdrop 0.0 \
--per_device_train_batch_size 30