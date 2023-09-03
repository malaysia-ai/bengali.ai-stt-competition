import datasets
import random
import pandas as pd
import torch
from transformers import AutoTokenizer
from typing import Mapping, Tuple
import librosa
# import en_core_web_sm


import librosa

class BengaliDataset(torch.utils.data.Dataset):
    def __init__(self, df, processor):
        self.df = df
        self.processor = processor


    def __getitem__(self, idx):
        audio_path = self.df.loc[idx]['path']
        audio_array = self.read_audio(audio_path)
        
        inputs = self.processor(
            audio_array,
            sampling_rate=16000,
            return_tensors='pt'  
        )
        
        with self.processor.as_target_processor():
            labels = self.processor(self.df.loc[idx]['sentence']).input_ids
        
        return {'input_values': inputs['input_values'][0], 'labels': labels}
        
    def __len__(self):
        return len(self.df)

    def read_audio(self, mp3_path):
        target_sr = 16000  # Set the target sampling rate
        
        audio, sr = librosa.load(mp3_path, sr=None)  # Load with original sampling rate
        audio_array = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        return audio_array
