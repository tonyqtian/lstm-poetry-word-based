import collections
import numpy as np
import re
import random


class TextProcessor(object):
    @staticmethod
    def from_file(input_file):
        with open(input_file, 'r', encoding = 'utf8') as fh:
            text = fh.read()
        return TextProcessor(text)

    def __init__(self, text, rawEng=False):
        if rawEng:
            self.words = self._text2words(text)
        else:
            self.words = [w for w in text.split()]
        self.id2word = None
        self.word2id = None
        self.vector = None

    def set_vocab(self, word2id):
        self.word2id = word2id
        return self

    def create_vocab(self, size):
        counter = collections.Counter(self.words)
        print( 'Vocabulary size reduced from %s to %s' % (len(counter), size) )
        count_pairs = counter.most_common(size-1)
        self.id2word = list(dict(count_pairs).keys())
        self.id2word[-1] = '<unk>'
        self.word2id = dict(zip(self.id2word, range(len(self.id2word))))

    def get_vector(self):
        unk = self.word2id['<unk>']
        self.vector = [self.word2id[word] if word in self.word2id else unk for word in self.words]
        return self.vector

    def save_converted(self, filename):
        with open(filename, 'w') as fh:
            for wid in self.vector:
                fh.write(self.id2word[wid]+' ')

    @staticmethod
    def _text2words(text):
        # prepare for word based processing
        re4 = re.compile(r'\.\.+')
        re5 = re.compile(r' +')

        text = text.lower()
        text = re4.sub(' <3dot> ', text)
        text = text.replace(',', ' , ')
        text = text.replace('.', ' . ')
        text = text.replace('/', ' . ')
        text = text.replace('(', ' ( ')
        text = text.replace(')', ' ) ')
        text = text.replace('[', ' ( ')
        text = text.replace(']', ' ) ')
        text = text.replace(':', ' : ')
        text = text.replace("'", " '")
        text = text.replace('?', ' ? ')
        text = text.replace(';', ' . ')
        text = text.replace('-', ' -')

        text = text.replace('<3dot>', ' ... ')
        text = text.replace('"', '')

        text = re5.sub(' ', text)
        text = text.replace('\n', ' <nl> ')
        return ['\n' if w == '<nl>' else w for w in text.split()]


def train_iterator(raw_data, batch_size, num_steps):
    raw_data = np.array(raw_data, dtype=np.int32)

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i * num_steps:(i + 1) * num_steps]
        y = data[:, i * num_steps + 1:(i + 1) * num_steps + 1]
        yield (x, y)
        
data_index = 0

def generate_batch(data, batch_size, num_skips, skip_window):
    
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1 # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [ skip_window ]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels