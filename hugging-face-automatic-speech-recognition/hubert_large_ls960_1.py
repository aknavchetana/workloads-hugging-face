# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Eo_7UzLmxCwpui6dV9VIZcZSoRS_10Uj
"""
from transformers import Wav2Vec2ForCTC
from transformers import Wav2Vec2Tokenizer

#from transformers import wav2vec2CTCTokenizer
from transformers import Wav2Vec2Processor, HubertForCTC
from datasets import load_dataset
import torch
import time

dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
dataset = dataset.sort("id")
sampling_rate = dataset.features["audio"].sampling_rate

processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")
#calculating model size
param_size = 0
buffer_size = 0
for param in model.parameters():
    param_size += param.nelement() * param.element_size()

for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

size_all_mb = (param_size + buffer_size) / 1024**2
print('Size: {:.3f} MB'.format(size_all_mb))

# audio file is decoded on the fly
inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
with torch.no_grad():
    start=time.process_time()
    logits = model(**inputs).logits
predicted_ids = torch.argmax(logits, dim=-1)

# transcribe speech
transcription = processor.batch_decode(predicted_ids)

print("execution time->",time.process_time()-start)
print("transcription->",transcription[0])


with processor.as_target_processor():
    inputs["labels"] = processor(dataset[0]["text"], return_tensors="pt").input_ids

# compute loss
loss = model(**inputs).loss
print("loss->",round(loss.item(), 2))




