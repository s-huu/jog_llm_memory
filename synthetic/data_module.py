import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import datasets
from utils import get_model_identifiers_from_yaml, add_dataset_index
    

def get_names(min_len=5, max_len=2000, batch_size=1, file="common_names_ft.txt"):
    data = []
    for line in open(file, "r"):
        raw_text = line
        raw_text = raw_text.strip()
        if len(raw_text) > min_len:
            data.append(str(raw_text[:max_len]))
    data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
    return data  

class TextDatasetNames(Dataset):
    def __init__(self, tokenizer, model_family, max_length=512, split = None, file=None):
        super(TextDatasetNames, self).__init__()
        print(file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = get_names(file=file)

        self.model_configs = get_model_identifiers_from_yaml(model_family)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        full_text = self.data[idx]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        converted_data = convert_data_to_model_format(full_text, self.tokenizer, self.max_length)
        pad_input_ids_list.append(converted_data[0])
        label_list.append(converted_data[1])
        pad_attention_mask_list.append(converted_data[2])


        return torch.stack(pad_input_ids_list).squeeze(),\
                torch.stack(label_list).squeeze(),\
                torch.stack(pad_attention_mask_list).squeeze()

class TextDatasetNamesForget(Dataset):
    def __init__(self, tokenizer, model_family, max_length=512, split = None, file=None):
        super(TextDatasetNamesForget, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        data = get_names(file=file)
        self.forget_data = [data[i] for i in [0,1,2,3,12,13,14,15]]
        self.retain_data = [data[i] for i in [4,5,6,7,8,9,10,11,16,17,18,19]]
        self.split1 = 'forget'
        self.split2 = 'retain'

        self.model_configs = get_model_identifiers_from_yaml(model_family)
    def __len__(self):
        # print(self.data)
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []
        for data_type in [self.split1, self.split2]:
            #use questions from forget set if split is idk or forget
            data = self.retain_data if data_type == "retain" else self.forget_data
            idx = idx if data_type != "retain" else (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)
            full_text = data[idx]
                
            converted_data = convert_data_to_model_format(full_text, self.tokenizer, self.max_length)
            rets.append(converted_data)
        return rets

    
def convert_data_to_model_format(data, tokenizer, max_length=500):
    encoded = tokenizer(
        data[0], 
        add_special_tokens=True, 
        max_length=5000, 
        truncation=True, 
    )
    pad_length = max_length - len(encoded.input_ids)
    # print(encoded['input_ids'])
    pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length

    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length-1)

    # print(pad_input_ids)
    # print(label)
    # print(pad_attention_mask)
    return torch.tensor(pad_input_ids),torch.tensor(label),torch.tensor(pad_attention_mask)

def collate_fn(batch):
    input_ids, attention_masks = zip(*batch)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=-100)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    return input_ids, attention_masks


def custom_data_collator(samples):
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)

def custom_data_collator_with_indices(samples):
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    indices = [s[3] for s in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask), torch.stack(indices)

def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1)

    return loss
