# Finetune MMS for Automatic Speech Recognition

Finetune mms-1b-all on Bengali speech recordings from [Bengali Speech Recognition Competition](https://www.kaggle.com/competitions/bengaliai-speech/data). The script uses custom torch dataset for finetuning, make changes to script based on your data.

## Finetune mms-1b-all
The following command shows how to fine-tune [mms-1b-all](https://huggingface.co/facebook/mms-1b-all) using `run_speech_recognition_ctc.py` script.

```
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
--do_train

```

## Finetune mms-1b-all with DeepSpeed
The following command shows how to fine-tune [mms-1b-all](https://huggingface.co/facebook/mms-1b-all) with DeepSpeed integration using `run_speech_recognition_ctc.py` script. Specify deepspeed configuration file path in your command.

```
~/.local/bin/deepspeed run_speech_recognition_ctc.py \
--deepspeed ds_config_zero3.json \
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
--do_train 
```

### NOTE

If you encounter problem with running the script when loading adapter weights `model.load_adapter({target_language})` as follows, save your pretrained model with fitted adapter weight first in local and pass saved model in model_name_or_path argument:

```
model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all")
model.load_adapter({target_language})
model.save_pretrained('./mms-1b-all')

```

## wav2vec2 with KenLM



## Inference

Refer notebook `inference.ipynb` for loading model checkpoint and perform inference with finetuned mms-1b model.

## Reference

- https://github.com/huggingface/transformers/tree/main/examples/pytorch/speech-recognition

