from transformers import AutoModelForCausalLM, AutoTokenizer
from data_module import custom_data_collator
import torch.nn as nn
import torch

from tqdm import tqdm

import os

import numpy as np
import random


hf_key = 'NousResearch/Llama-2-7b-chat-hf'
tokenizer = AutoTokenizer.from_pretrained(hf_key)
batch_size = 4

# Tokenize the sentences
def preprocess_batch(examples):
    return tokenizer(examples, padding=True, truncation=True, return_tensors='pt')

open_token='[INST]'
close_token='[/INST]'

def run_single_generation(str_inputs):
    tokenized_inputs = tokenizer(str_inputs, return_tensors="pt").to(0)
    model.eval()
    generation_output = model.generate(tokenized_inputs.input_ids, return_dict_in_generate=True, output_scores=True, max_length=100, num_beams=1, temperature=1, do_sample=False)

    input_length = len(tokenized_inputs[0])
    generated_tokens = generation_output.sequences[:, input_length:]
    labels = tokenized_inputs

    for seq in generation_output[0]:
        dec = tokenizer.decode(seq, skip_special_tokens=True)
        return dec

names_list = ["James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph",\
               "Charles", "Thomas", "Christopher", "Daniel", "Matthew", "Kevin", "Brian",\
               "George", "Edward", "Ronald", "Timothy", "Jason", "Jeffrey", "Ryan", "Jacob",\
               "Gary", "Raymond", "Alexander", "Patrick", "Jack", "Noah", "Gerald", "Carl",\
               "Terry", "Sean", "Arthur", "Austin", "Alec", "Clyde"]

num_names = len(names_list)

checkpoint = f"shengyuanhu/common_names_7_repetition_relearn_8_17"
model = AutoModelForCausalLM.from_pretrained(checkpoint).cuda(0)
avg_loss = {0:[],1:[]}
acc = 0
for i in range(100):
  random_len = random.randrange(num_names)
  selected = random.sample(names_list,max(random_len,1))
  query = ' '.join(selected)
  tokenized_inputs_for_gen = tokenizer(query + ' Anthony', return_tensors="pt").to(0)
  generation_output = model.generate(tokenized_inputs_for_gen.input_ids, return_dict_in_generate=True, output_scores=True, max_length=100, num_beams=1, temperature=1, do_sample=False)
  dec = tokenizer.decode(generation_output[0][0], skip_special_tokens=True)
  if "Anthony Mark" in dec:
    acc += 1

  query = query + ' Anthony Mark'
  tokenized_inputs = tokenizer(query, return_tensors="pt").to(0)
  labels = tokenized_inputs
  loss = nn.CrossEntropyLoss(reduction='none')(model(**tokenized_inputs).logits[0],labels.input_ids[0])
  avg_loss[0].append(loss[-2].item())
  avg_loss[1].append(loss[-1].item())
print('Attack Success Rate: ', acc)
print('Avrage Loss: ', avg_loss)
