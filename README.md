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

## Dataset Cleaning
We conducted data cleaning on the audio transcription by:

- Removing all kinds of punctuation. Manually added quotations into the list.
```
import string  
base = string.punctuation 
punct = base + '“”'
punct
>>> !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~“”
```

- Strip any whitespaces at the front and at the back of string.
- Normalize the transcriptions using bnunicodenormalizer.
```
!pip install bnunicodenormalizer 

from bnunicodenormalizer import Normalizer 
bnorm=Normalizer()

def norm(transcription):
  text_list = []
  texts = transcription.split()

  for text in texts:
    result = bnorm(text)
    if len(result["ops"]) > 0:
      text_list.append(result["ops"][0]["after"])
    else:
      text_list.append(text)

  normalized_transcription = ", ".join(text_list)
```
In order for the normalizer to work on our data, we had to split it into tokens (words). This is due to the normalizer not accepting a full sentence and can only process single words. We also noticed that the process could take some time. Hence, we implemented multi-threading to speed up the entire normalization process. We applied the data cleaning process on the base dataset from Bengali.AI and also Google Fleurs.

- Calculated the length for each of the audio files.

Based on our analysis, we noticed the audio files had varying lengths. This resulted in unexpected OOM during the validation. Thus, we filtered the audio files and only took 18~20 seconds. 


- Mixing Training and Validation Data

The data given by Bengali.AI was already separated into 2 (training and validation). We noticed that the validation data was somewhat better as compared to the training data. Thus, we mixed some of the validation data into the training set. 


- [Untested Process]

In total we had almost 1 million audio files. From our analysis and preliminary training evaluation, we noticed that the data consists of very bad audio quality. This affected the performance of the model directly and prevented the model to improve. Initially we thought of calculating the amplitude of the audio files. But, determining the amplitude does not correlate to background noise. What we thought of doing is to filter the audio files that the model has trouble to transcribe. 

To put it simply, by training the model on the cleaned data, we would use the best checkpoint and transcribe the training and validation data. The data that produces the worst WER (word error rate) shall be excluded. This is to ensure that the model is only trained and validated on good quality data. 


- Model Saturation

In the early stages of our training phase, we dumped the entire dataset onto the model. Not only did we face OOM problems, but we slowly notice that the model became saturated. Due to not having an early stopping function and not stopping the training at certain points, the model was performing even worst as compared to those who only train with 10%-20% data. 

To prevent from the model being saturated, we had to feed the data into byte-sized pieces (pun intended). As it turns out, feeding a small partition of the data improved the model' performance quite drastically. From there, we tested the model using the test set. If the performance is increasing, we maintain the current data partition. If the performance is constant or decreasing, we would increase the data partition, i.e. add in more data. 

By using this strategy, we were able to prevent the model from being saturated. The only downside is that we had to keep track of the model's performance and constantly push out the checkpoints. 


All in all, implementing these strategies showed a significant improvement on the model's performance as compared to not cleaning the data at all. In our previous projects, we would only conduct extremely minor data cleaning process as to preserve the structure of the data. We wanted the model to learn the clean and also unclean data in the hopes of making it robust. Unfortunately, this was not the case. 

Not cleaning the data at all would degrade the performance and training the model for long periods of time will result in model saturation.

## Language Modeling with KenLM
- Install KenLM as mentioned in install-kenlm.sh.
- Refer to the notebook kenlm.ipynb for creating a KenLM language model for the wav2vec2 model.

### Overview
KenLM is a fast language model and accurate. We use n-gram LM to enhance asr model.

### Summary
In this notebook, we train the models with the texts from competition datasets (train and validation) and google fleurs bengali.

We attached KenLM language model into our wav2vec model to improve our model score and use Word Error Rate (WER) and Character Error Rate (CER) as our evaluation metrics benchmark the predictions.

After rigorous evaluations, 3-gram showed the best performance among 4-gram and 5-gram models.

NOTE: For KenLM, we only used the cleaned text. This is to ensure that the model is able to decode properly and match with the cleaned transcriptions. 

## Inference
To perform inference with the fine-tuned model, refer to the notebook inference.ipynb for loading the model checkpoint and conducting inference.

## Solution Overview for Bengali.AI STT (Bronze Medal)
This section outlines the key steps that led to our placement in top 8% in the Bengali.AI Speech-to-Text (STT) competition, including dataset cleaning, model selection, and post-processing.

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
