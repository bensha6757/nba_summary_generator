import torch
import random
import json


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 n_descriptions=50):
        self.data = data
        self.n_descriptions = n_descriptions

    def __len__(self):
        return len(self.data)

    def get_summary(self, example):
        return example['summary'] + ' </s>'

    def __getitem__(self, index):
        example = self.data[index] # here
        summary = self.get_summary(example)
        descriptions = example['descriptions'][:self.n_descriptions]

        return {
            'index': index,
            'summary': summary,
            'descriptions': descriptions
        }

    def get_example(self, index):
        return self.data[index]


def encode_descriptions(batch_text_descriptions, tokenizer, max_length):
    description_ids, description_masks = [], []
    for k, text_descriptions in enumerate(batch_text_descriptions):
        p = tokenizer.batch_encode_plus(
            text_descriptions,
            max_length=max_length,
            pad_to_max_length=True,
            return_tensors='pt',
            truncation=True
        )
        description_ids.append(p['input_ids'][None]) 
        description_masks.append(p['attention_mask'][None])

    description_ids = torch.cat(description_ids, dim=0)
    description_masks = torch.cat(description_masks, dim=0)
    return description_ids, description_masks.bool()


class Collator(object):
    def __init__(self, text_maxlength, tokenizer, answer_maxlength=0):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength

    def __call__(self, batch):
        assert (batch[0]['summary'] is not None)
        index = torch.tensor([example['index'] for example in batch])
        summary = [example['summary'] for example in batch]
        summary = self.tokenizer.batch_encode_plus(
            summary,
            max_length=self.answer_maxlength if self.answer_maxlength > 0 else None,
            pad_to_max_length=True,
            return_tensors='pt',
            truncation=True if self.answer_maxlength > 0 else False,
        )
        summary_ids = summary["input_ids"]
        summary_mask = summary["attention_mask"].bool()
        summary_ids = summary_ids.masked_fill(~summary_mask, -100)

        text_descriptions = [example['descriptions'] for example in batch]
        description_ids, description_masks = encode_descriptions(text_descriptions,
                                                     self.tokenizer,
                                                     self.text_maxlength)

        return (index, summary_ids, summary_mask, description_ids, description_masks)


def load_data(data_path=None, global_rank=-1, world_size=-1):
    assert data_path
    with open(data_path, 'r') as fin:
        data = json.load(fin)
    return [example for k, example in enumerate(data) if global_rank <= -1 or k % world_size == global_rank]
