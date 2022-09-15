import gzip
import json
import os
import random

import numpy as np
import torch

from torch.utils.data import Dataset


class BeerData(Dataset):
    def __init__(self, data_dir, aspect, mode, word2idx, balance=False, max_length=256, neg_thres=0.4, pos_thres=0.6,
                 stem='reviews.aspect{}.{}.txt.gz'):
        super().__init__()
        self.mode_to_name = {'train': 'train', 'dev': 'heldout'}
        self.mode = mode
        self.neg_thres = neg_thres
        self.pos_thres = pos_thres
        self.input_file = os.path.join(data_dir, stem.format(str(aspect), self.mode_to_name[mode]))
        self.inputs = []
        self.masks = []
        self.labels = []
        self._convert_examples_to_arrays(
            self._create_examples(aspect, balance), max_length, word2idx)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        inputs, masks, labels = self.inputs[item], self.masks[item], self.labels[item]
        return inputs, masks, labels

    def _create_examples(self, aspect, balance=False):
        examples = []
        with gzip.open(self.input_file, "rt") as f:
            lines = f.readlines()
            for (i, line) in enumerate(lines):
                labels, text = line.split('\t')
                labels = [float(v) for v in labels.split()]
                if labels[aspect] <= self.neg_thres:
                    label = 0
                elif labels[aspect] >= self.pos_thres:
                    label = 1
                else:
                    continue
                examples.append({'text': text, "label": label})
        print('Dataset: Beer Review')
        print('{} samples has {}'.format(self.mode_to_name[self.mode], len(examples)))

        pos_examples = [example for example in examples if example['label'] == 1]
        neg_examples = [example for example in examples if example['label'] == 0]

        print('%s data: %d positive examples, %d negative examples.' %
              (self.mode_to_name[self.mode], len(pos_examples), len(neg_examples)))

        if balance:

            random.seed(12252018)

            print('Make the Training dataset class balanced.')

            min_examples = min(len(pos_examples), len(neg_examples))

            if len(pos_examples) > min_examples:
                pos_examples = random.sample(pos_examples, min_examples)

            if len(neg_examples) > min_examples:
                neg_examples = random.sample(neg_examples, min_examples)

            assert (len(pos_examples) == len(neg_examples))
            examples = pos_examples + neg_examples
            print(
                'After balance training data: %d positive examples, %d negative examples.'
                % (len(pos_examples), len(neg_examples)))
        return examples

    def _convert_single_text(self, text, max_length, word2idx):
        """
        Converts a single text into a list of ids with mask.
        """
        input_ids = []

        text_ = text.strip().split(" ")

        if len(text_) > max_length:
            text_ = text_[0:max_length]

        for word in text_:
            word = word.strip()
            try:
                input_ids.append(word2idx[word])
            except:
                # if the word is not exist in word2idx, use <unknown> token
                input_ids.append(0)

        # The mask has 1 for real tokens and 0 for padding tokens.
        input_mask = [1] * len(input_ids)

        # zero-pad up to the max_seq_length.
        while len(input_ids) < max_length:
            input_ids.append(0)
            input_mask.append(0)

        assert len(input_ids) == max_length
        assert len(input_mask) == max_length

        return input_ids, input_mask

    def _convert_examples_to_arrays(self, examples, max_length, word2idx):
        """
        Convert a set of train/dev examples numpy arrays.
        Outputs:
            data -- (num_examples, max_seq_length).
            masks -- (num_examples, max_seq_length).
            labels -- (num_examples, num_classes) in a one-hot format.
        """

        data = []
        labels = []
        masks = []
        for example in examples:
            input_ids, input_mask = self._convert_single_text(example["text"],
                                                              max_length, word2idx)

            data.append(input_ids)
            masks.append(input_mask)
            labels.append(example["label"])

        self.inputs = torch.from_numpy(np.array(data))
        self.masks = torch.from_numpy(np.array(masks))
        self.labels = torch.from_numpy(np.array(labels))


class BeerAnnotation(Dataset):

    def __init__(self, annotation_path, aspect, word2idx, max_length=256, neg_thres=0.4, pos_thres=0.6):
        super().__init__()
        self.inputs = []
        self.masks = []
        self.labels = []
        self.rationales = []
        self._create_example(annotation_path, aspect, word2idx, max_length, pos_thres, neg_thres)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        inputs, masks, labels, rationales = self.inputs[item], self.masks[item], self.labels[item], self.rationales[
            item]
        return inputs, masks, labels, rationales

    def _create_example(self, annotation_path, aspect, word2idx, max_length, pos_thres, neg_thres):
        data = []
        masks = []
        labels = []
        rationales = []

        print('Dataset: Beer Review')

        with open(annotation_path, "rt", encoding='utf-8') as fin:
            for counter, line in enumerate(fin):
                item = json.loads(line)

                # obtain the data
                text_ = item["x"]
                y = item["y"][aspect]
                rationale = item[str(aspect)]

                # check if the rationale is all zero
                if len(rationale) == 0:
                    # no rationale for this aspect
                    continue

                # process the label
                if float(y) >= pos_thres:
                    y = 1
                elif float(y) <= neg_thres:
                    y = 0
                else:
                    continue

                # process the text
                input_ids = []
                if len(text_) > max_length:
                    text_ = text_[0:max_length]

                for word in text_:
                    word = word.strip()
                    try:
                        input_ids.append(word2idx[word])
                    except:
                        # word is not exist in word2idx, use <unknown> token
                        input_ids.append(0)

                # process mask
                # The mask has 1 for real word and 0 for padding tokens.
                input_mask = [1] * len(input_ids)

                # zero-pad up to the max_seq_length.
                while len(input_ids) < max_length:
                    input_ids.append(0)
                    input_mask.append(0)

                assert (len(input_ids) == max_length)
                assert (len(input_mask) == max_length)

                # construct rationale
                binary_rationale = [0] * len(input_ids)
                for zs in rationale:
                    start = zs[0]
                    end = zs[1]
                    if start >= max_length:
                        continue
                    if end >= max_length:
                        end = max_length

                    for idx in range(start, end):
                        binary_rationale[idx] = 1

                data.append(input_ids)
                labels.append(y)
                masks.append(input_mask)
                rationales.append(binary_rationale)

        self.inputs = torch.from_numpy(np.array(data))
        self.labels = torch.from_numpy(np.array(labels))
        self.masks = torch.from_numpy(np.array(masks))
        self.rationales = torch.from_numpy(np.array(rationales))
        tot = self.labels.shape[0]
        print('annotation samples has {}'.format(tot))
        pos = torch.sum(self.labels)
        neg = tot - pos
        print('annotation data: %d positive examples, %d negative examples.' %
              (pos, neg))
