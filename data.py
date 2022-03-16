import torch
import random
import json


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 n_descriptions=5):
        self.data = data
        self.n_descriptions = n_descriptions
        self.sort_data() # here

    def __len__(self):
        return len(self.data)

    def get_summary(self, example):
        return example['summary'] + ' </s>'

    def __getitem__(self, index):
        example = self.data[index] # here
        summary = self.get_summary(example)
        descriptions = example['descriptions'][:self.n_descriptions]

        scores = [float(c['score']) for c in descriptions]
        scores = torch.tensor(scores)

        return {
            'index': index,
            'summary': summary,
            'descriptions': descriptions,
            'scores': scores # here
        }

    def sort_data(self): # here
        if self.n_descriptions is None or not 'score' in self.data[0]['descriptions'][0]:
            return
        for ex in self.data:
            ex['descriptions'].sort(key=lambda x: float(x['score']), reverse=True)

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
    if data_path.endswith('.jsonl'):
        data = open(data_path, 'r')
    elif data_path.endswith('.json'):
        with open(data_path, 'r') as fin:
            data = json.load(fin)
    examples = []
    for k, example in enumerate(data):
        if global_rank > -1 and not k % world_size == global_rank:
            continue
        if data_path is not None and data_path.endswith('.jsonl'):
            example = json.loads(example)
        if not 'id' in example:
            example['id'] = k
        for c in example['descriptions']:
            if not 'score' in c:
                c['score'] = 1.0 / (k + 1)
        examples.append(example)
    ## egrave: is this needed?
    if data_path is not None and data_path.endswith('.jsonl'):
        data.close()

    return examples


class RetrieverCollator(object):
    def __init__(self, tokenizer, description_maxlength=200, question_maxlength=40):
        self.tokenizer = tokenizer
        self.description_maxlength = description_maxlength
        self.question_maxlength = question_maxlength

    def __call__(self, batch):
        index = torch.tensor([ex['index'] for ex in batch])

        question = [ex['question'] for ex in batch]
        question = self.tokenizer.batch_encode_plus(
            question,
            pad_to_max_length=True,
            return_tensors="pt",
            max_length=self.question_maxlength,
            truncation=True
        )
        question_ids = question['input_ids']
        question_mask = question['attention_mask'].bool()

        if batch[0]['scores'] is None or batch[0]['descriptions'] is None:
            return index, question_ids, question_mask, None, None, None

        scores = [ex['scores'] for ex in batch]
        scores = torch.stack(scores, dim=0)

        descriptions = [ex['descriptions'] for ex in batch]
        description_ids, description_masks = encode_descriptions(
            descriptions,
            self.tokenizer,
            self.description_maxlength
        )

        return (index, question_ids, question_mask, description_ids, description_masks, scores)


class TextDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 title_prefix='title:',
                 description_prefix='context:'):
        self.data = data
        self.title_prefix = title_prefix
        self.description_prefix = description_prefix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        text = self.title_prefix + " " + example[2] + " " + \
               self.description_prefix + " " + example[1]
        return example[0], text


class TextCollator(object):
    def __init__(self, tokenizer, maxlength=200):
        self.tokenizer = tokenizer
        self.maxlength = maxlength

    def __call__(self, batch):
        index = [x[0] for x in batch]
        encoded_batch = self.tokenizer.batch_encode_plus(
            [x[1] for x in batch],
            pad_to_max_length=True,
            return_tensors="pt",
            max_length=self.maxlength,
            truncation=True
        )
        text_ids = encoded_batch['input_ids']
        text_mask = encoded_batch['attention_mask'].bool()

        return index, text_ids, text_mask
