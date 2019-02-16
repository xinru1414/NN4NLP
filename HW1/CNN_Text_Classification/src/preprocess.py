from typing import List, Optional, Tuple
from tqdm import tqdm
import config
from sklearn.model_selection import KFold
import numpy as np


class Examples:
    def __init__(self, sents, labels):
        self.sents = sents
        self.labels = labels

        assert len(self.sents) == len(self.labels), "There must be the same number of sents as labels"

    def __add__(self, other: 'Examples'):
        assert isinstance(other, Examples), f'You can only add two example together not {type(other)} and Example'
        return Examples(self.sents+other.sents, self.labels+other.labels)

    def __len__(self):
        return len(self.sents)

    def __iter__(self):
        return iter([self.sents, self.labels])

    def shuffled(self):
        c = list(zip(self.sents, self.labels))
        np.random.shuffle(c)
        return Examples(*zip(*c))

    def get_sents(self, indexes: Optional[List[int]] = None):
        if indexes is None:
            return self.sents
        else:
            return [x for i, x in enumerate(self.sents) if i in indexes]

    def get_labels(self, indexes: Optional[List[int]] = None):
        if indexes is None:
            return self.labels
        else:
            return [x for i, x in enumerate(self.labels) if i in indexes]


class DataLoader:
    def __init__(self, cfg):
        self.config = cfg

        self.UNK = '<UNK>'
        self.PAD = '<PAD>'
        self.k = config.k

        self.train_file_path = config.train_file_path
        self.dev_file_path = config.dev_file_path
        self.test_file_path = config.test_file_path

        print('building vocabularies from all train, dev and test')
        self.w2i, self.i2w, self.l2i, self.i2l = self.build_vocabs(self.train_file_path, self.dev_file_path, self.test_file_path)
        self.UNK_IDX = self.w2i[self.UNK]
        self.PAD_IDX = self.w2i[self.PAD]

        self.vocab_size = len(self.w2i)
        self.label_size = len(self.l2i)
        print(f'vocab size is {self.vocab_size}, label size is {self.label_size}')

        print('train and eval on train + dev')
        train_examples = self.load_data(self.train_file_path)
        dev_examples = self.load_data(self.dev_file_path)

        full_training_examples = train_examples + dev_examples

        print('padding')
        self.dev_examples = self.pad_sentences(dev_examples)
        self.train_examples = self.pad_sentences(train_examples)
        self.full_training_examples = self.pad_sentences(full_training_examples)
        print('test on dev')
        self.test_examples = self.pad_sentences(self.load_data(self.dev_file_path))
        # print('test on test')
        self.test_examples2 = self.pad_sentences(self.load_data(self.test_file_path))

        print('reading in examples from train and k folding')
        #self.folds = [(self.train_examples, self.dev_examples)]
        self.folds = self.k_fold(examples=self.full_training_examples, k=self.k)

    def build_vocabs(self, *file_paths):
        words_set = set()
        labels = set()
        for file in file_paths:
            with open(file, 'r') as f:
                for line in f:
                    label, words = line.strip().split('|||')
                    labels.add(label.strip())
                    for word in words.strip().split():
                        words_set.add(word)

        word_list = list(words_set)
        w2i = {word: i for i, word in enumerate(word_list)}
        w2i[self.UNK] = len(w2i)
        w2i[self.PAD] = len(w2i)
        i2w = {i: word for word, i in w2i.items()}

        l2i = {label: i for i, label in enumerate(list(labels))}
        i2l = {i: label for label, i in l2i.items()}

        return w2i, i2w, l2i, i2l

    def load_data(self, *file_paths) -> Examples:
        labels = []
        sents = []
        for file in file_paths:
            with open(file, 'r') as f:
                for line in f:
                    label, words = line.strip().split('|||')
                    if label == 'UNK':
                        labels.append(-1)
                    else:
                        labels.append(self.l2i[label.strip()])
                    sent = []
                    for word in words.strip().split():
                        sent.append(self.w2i[word] if word in self.w2i else self.UNK_IDX)
                    sents.append(sent)
            assert len(sents) == len(labels)
        return Examples(sents=sents, labels=labels)

    def pad_sentences(self, examples:Examples) -> Examples:
        labels = examples.labels
        sents = examples.sents
        max_len = max(len(sent) for sent in sents)
        padded_sents = [sent if len(sent) == max_len else sent + [self.PAD_IDX] * (max_len - len(sent)) for sent in sents]
        assert [len(sent) == max_len for sent in sents]
        return Examples(sents=padded_sents, labels=labels)

    def k_fold(self, examples: Examples, k: int) -> List[Tuple[Examples, Examples]]:
        kf = KFold(n_splits=k, shuffle=True)
        return_value = []
        for train_index, test_index in tqdm(kf.split(range(len(examples))), total=k):
            X_train, X_test = examples.get_sents(train_index), examples.get_sents(test_index)
            y_train, y_test = examples.get_labels(train_index), examples.get_labels(test_index)
            return_value += [(Examples(X_train, y_train), Examples(X_test, y_test))]
        return return_value


def batch(examples: Examples, batch_size: int):
    batches = []
    batch_num = int(np.ceil(len(examples) / float(batch_size)))
    for i in range(batch_num):
        batch_sents = examples.sents[i*batch_size:(i+1)*batch_size]
        batch_labels = examples.labels[i*batch_size:(i+1)*batch_size]
        batch_data = Examples(batch_sents, batch_labels)
        batches.append(batch_data)

    for b in batches:
        yield b


def load_pte(pte, frequent):
    '''
    load pte from a text file.
    :param pte: path to pte
    :param frequent: most frequent n embedding
    :return: embedding, w2i
    '''
    vectors = []
    w2i = {}

    emb_dim = 300
    with open(pte, 'r', encoding='utf-8') as f:
        for i, row in enumerate(f):
            if i == 0:
                # no need to read first line
                pass
            else:
                word, vec = row.rstrip().split(' ', 1)
                word = word.lower()
                vec = np.fromstring(vec, sep=' ')
                if word not in w2i:
                    assert vec.shape == (emb_dim,), i
                    w2i[word] = len(w2i)
                    vectors.append(vec[None])
                else:
                    pass
            # only load the most frequent
            if len(w2i) >= frequent:
                break

    assert len(w2i) == len(vectors)

    embeddings = np.concatenate(vectors, 0)
    return embeddings, w2i