### Dataset Cleaning
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