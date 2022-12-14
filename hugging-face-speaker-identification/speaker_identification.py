# -*- coding: utf-8 -*-
"""Hubert-Large for Speaker Identification

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1N86Hmz1AaRN8gEn9xmFF1xljJlf6RA6k
"""

import torch
import librosa
from datasets import load_dataset
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor

def map_to_array(example):
    speech, _ = librosa.load(example["file"], sr=16000, mono=True)
    example["speech"] = speech
    return example

# load a demo dataset and read audio files
dataset = load_dataset("anton-l/superb_demo", "si", split="test")
dataset = dataset.map(map_to_array)

model = HubertForSequenceClassification.from_pretrained("superb/hubert-large-superb-sid")
#calculating model size 
param_size = 0
buffer_size = 0
for param in model.parameters():
    param_size += param.nelement() * param.element_size()

for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

size_all_mb = (param_size + buffer_size) / 1024**2
print('Size: {:.3f} MB'.format(size_all_mb))

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/hubert-large-superb-sid")

# compute attention masks and normalize the waveform if needed
inputs = feature_extractor(dataset[:2]["speech"], sampling_rate=16000, padding=True, return_tensors="pt")


#calculating time of execution of model
import time
start=time.process_time()

logits = model(**inputs).logits
predicted_ids = torch.argmax(logits, dim=-1)

#printing execution time
print("execution time of model")
print(time.process_time()-start)



labels = [model.config.id2label[_id] for _id in predicted_ids.tolist()]

import numpy as np
actual = []
for i in range(len(dataset[:2])):
    actual.append(dataset[i]["label"])
actual = np.array(actual)
actual=actual[:2]
from sklearn.metrics import confusion_matrix , classification_report

import torch

print("Classification Report: \n", classification_report(actual, np.array(predicted_ids)))
