import os
import json
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from dataset import BengaliDataset
from transformers import Trainer,AutoProcessor
from datasets import load_metric, Audio
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor
) 
# import wandb
import librosa
import re
import warnings
warnings.simplefilter('ignore')
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from transformers import TrainingArguments


## Tokenizer? - TO CHANGE
def save_vocab(dataframe):
    """
    Saves the processed vocab file as 'vocab.json', to be ingested by tokenizer
    """
    vocab = construct_vocab(dataframe['sentence'].tolist())
    vocab_dict = {v: k for k, v in enumerate(vocab)}
    vocab_dict["__"] = vocab_dict[" "]
    _ = vocab_dict.pop(" ")
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)


### Padding (To make all embeddings to have the same length)
def ctc_data_collator(batch):
    """
    data collator function to dynamically pad the data
    """
    input_features = [{"input_values": sample["input_values"]} for sample in batch]
    label_features = [{"input_ids": sample["labels"]} for sample in batch]
    batch = processor.pad(
        input_features,
        padding=True,
        return_tensors="pt",
    )
    with processor.as_target_processor():
        labels_batch = processor.pad(
            label_features,
            padding=True,
            return_tensors="pt",
        )
        
    labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
    batch["labels"] = labels
    return batch


# TO CHANGE
def construct_vocab(texts):
    """
    Get unique characters from all the text in a list
    """
    all_text = " ".join(texts)
    vocab = list(set(all_text))
    return vocab
    
### Data cleaning, remove punctuations and lowercase
def remove_special_characters(data):

    chars_to_ignore_regex = ', ? . ! - \; \: \" “ % ” �'
    
    data["sentence"] = re.sub(chars_to_ignore_regex, "", data[text_column_name]).lower() + " "
  
    return data


### Word Error Rate (Evaluation Metrics)
def compute_metrics(pred):

    wer_metric = load_metric("wer")

    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}



if __name__ == "__main__":

    output_dir = './mms-1b-all'

    model_name = 'facebook/mms-1b-all'

    df = pd.read_csv("/home/ubuntu/bengali/data/train.csv")

    df['path'] = df['id'].apply(lambda x: os.path.join('/home/ubuntu/bengali/data/train_mp3s', x+'.mp3'))


    train = df[df['split'] == 'train'].sample(frac=.2).reset_index(drop=True)
    val = df[df['split'] == 'valid'].reset_index(drop=True)
    print(f"Training on samples: {len(train)}, Validation on samples: {len(val)}")


    # ### wav2vec2 does not have a decoder (Need to create vocab file?)
    # save_vocab(df)

   
    processor = AutoProcessor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)


    with open('vocab.json', 'w') as fopen:
        json.dump(processor.tokenizer.vocab['ben'], fopen)

    tokenizer = Wav2Vec2CTCTokenizer(
        "vocab.json", 
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="__"
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, 
        sampling_rate=16000, 
        padding_value=0.0, 
        do_normalize=True, 
        return_attention_mask=False
    )

    # model = Wav2Vec2ForCTC.from_pretrained(
    #     model_name,
    #     ctc_loss_reduction="mean", 
    #     pad_token_id=processor.tokenizer.pad_token_id
    #     )


    train_ds = BengaliDataset(train,processor)
    valid_ds = BengaliDataset(val,processor)


    """
       The first component of XLS-R consists of a stack of CNN layers that are used to extract acoustically meaningful
        - but contextually independent - features from the raw speech signal. This part of the model has already been 
        sufficiently trained during pretraining and as stated in the paper does not need to be fine-tuned anymore. 
    """
    model.freeze_feature_encoder()
    model.to('cuda')

    # AISYAH, HUSEIN SURUH LETAK DO_EVAL SO TAK MAKAN VRAM?
    training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=11,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=4,
            evaluation_strategy="steps",
            num_train_epochs=1,
            gradient_checkpointing=True,
            fp16=True,
            save_steps=200,
            eval_steps=200,
            logging_steps=100,
            learning_rate=3e-4,
            save_total_limit=2,
            do_eval=False  
             )

    trainer = Trainer(
        model=model,
        data_collator=ctc_data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_ds,   
        eval_dataset=valid_ds,    
        tokenizer=processor.feature_extractor,
    )



    checkpoint = None
    last_checkpoint = None
    # Detecting last checkpoint.
    if os.path.isdir(output_dir):
        last_checkpoint = get_last_checkpoint(output_dir)
        if last_checkpoint is not None:
            logging.info(
                f"Checkpoint detected, resuming training at {last_checkpoint} "
              
            )
            checkpoint = last_checkpoint

    trainer.train(resume_from_checkpoint=checkpoint)


    