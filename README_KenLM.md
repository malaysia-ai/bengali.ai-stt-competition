# KenLM

## Training Summary

### Overview
KenLM is a fast language model and accurate. We use n-gram LM to enhance our speech recognition system, in this case KenLM. 

### Summary:

In this [notebook](https://github.com/malaysia-ai/bengali.ai-stt-competition/blob/main/kenlm.ipynb), we train the models with the texts from competition datasets (train and validation) and google fleurs bengali.

We attached KenLM language model into our wav2vec model to do evaluation and use Word Error Rate (WER) and Character Error Rate (CER) as our evaluation metrics benchmark the predictions.

After rigorous evaluations, 3-gram showed the best performance among 4-gram and 5-gram models.




