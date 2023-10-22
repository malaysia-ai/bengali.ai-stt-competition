CUDA_VISIBLE_DEVICES='1' WANDB_DISABLED=true \
python3 trainer.py \
--model conformer_rnnt_base \
--output_dir conformer_rnnt_base \
--learning_rate="2e-5" \
--save_total_limit 3 \
--save_steps 1000 \
--eval_steps 1000 \
--logging_steps 10 \
--num_train_epochs 5 \
--per_device_train_batch_size 30