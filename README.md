# Automatic Speech Recognition

Finetune ASR Models on Bengali speech recordings from [Bengali Speech Recognition Competition](https://www.kaggle.com/competitions/bengaliai-speech/data). The script uses custom torch dataset for finetuning, make changes to script based on your data.

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

## Finetune Wav2vec2

```
~/.local/bin/deepspeed wav2vec2-finetune.py \
--model_name_or_path="ai4bharat/indicwav2vec_v1_bengali" \
--deepspeed ds_config_zero3.json \
--output_dir="./wav2vec2-bengali-xlsr-indi" \
--num_train_epochs="10" \
--per_device_train_batch_size="5" \
--learning_rate="5e-5" \
--evaluation_strategy="steps" \
--save_steps="1000" \
--eval_steps="1000" \
--logging_steps="50" \
--layerdrop="0.0" \
--save_total_limit="2" \
--gradient_checkpointing \
--do_train
```

## Training Scripts
Refer training folder for our training scripts used in this competition.

### NOTE

If you encounter problem with running the script when loading adapter weights `model.load_adapter({target_language})` as follows, save your pretrained model with fitted adapter weight first in local and pass saved model in model_name_or_path argument:

```
model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all")
model.load_adapter({target_language})
model.save_pretrained('./mms-1b-all')

```

## Create folds from dataset
We experimented with training models with different folds of data. Refer `create-fold.ipynb` for code on creating folds out of our dataset.


## Language Modeling with KenLM
- Install KenLM as mentioned in install-kenlm.sh.
- Refer to the notebook kenlm.ipynb for creating a KenLM language model for the wav2vec2 model.

## Inference
To perform inference with the fine-tuned model, refer to the notebook inference.ipynb for loading the model checkpoint and conducting inference.

## Solution Overview for Bengali.AI STT (Bronze Medal)
This section outlines the key steps that led to our placement in top 8% in the Bengali.AI Speech-to-Text (STT) competition, including dataset cleaning, model selection, and post-processing.

### Dataset Cleaning
Dataset normalization using bnmnormalizer significantly improved performance. Cleaned data  outperformed unnormalized and preprocessed data.

### Model Selection
ASR Model: The indicwav2vec model from ai4bharat was the most powerful model used in the competition.

Language Model: A 3-gram Language Model built with KenLM using the competition dataset text. 3-grams performed better than 4-grams and 5-grams.

### Post-Processing with BNMNormalizer
The output predictions from the ASR model and KenLM were processed using the BNMNormalizer package before submission.

### Not Working for Our Team

- xlsr-large-53: Although this model performed well on the competition dataset, it performed worst as the model was trained longer.

- conformer-rnn-t: The conformer model performed poorly even after extended training hours, so it was not pursued further.

- whisper-small: All wav2vec models outperformed the whisper model due to limited resources, hence we exclude whisper model from further experiments.

- External Dataset (google fleur): The external dataset did not contribute significantly to our results.

### Key Takeaways

- The Bengali AI Dataset contains a significant amount of poorly recorded audios. Filtering out low-quality data with high Word Error Rate (WER) can help improve model performance. We didn't try this method during the competition.
- Augmentation is crucial for increasing the amount of available data and enhancing model robustness.
- Save steps and evaluation dataset size plays a signifficant role to our model performance, having lower save steps value and smaller evaluation dataset produce poor models as compared to having large evaluation dataset.

## Acknowledgements
Thanks to @huseinzol05 for guiding the team on our first stt experience and providing us with GPU resources to train the model
