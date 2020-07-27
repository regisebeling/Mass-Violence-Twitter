import numpy as np
import re
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import load_files


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def get_datasets_20newsgroup(subset='train', categories=None, shuffle=True, random_state=42):
    """
    Retrieve data from 20 newsgroups
    :param subset: train, test or all
    :param categories: List of newsgroup name
    :param shuffle: shuffle the list or not
    :param random_state: seed integer to shuffle the dataset
    :return: data and labels of the newsgroup
    """
    datasets = fetch_20newsgroups(subset=subset, categories=categories, shuffle=shuffle, random_state=random_state)
    return datasets


def get_datasets_mrpolarity(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]

    datasets = dict()
    datasets['data'] = positive_examples + negative_examples
    target = [0 for x in positive_examples] + [1 for x in negative_examples]
    datasets['target'] = target
    datasets['target_names'] = ['positive_examples', 'negative_examples']
    return datasets


def get_datasets_localdata(container_path=None, categories=None, load_content=True,
                       encoding='utf-8', shuffle=True, random_state=42):
    """
    Load text files with categories as subfolder names.
    Individual samples are assumed to be files stored a two levels folder structure.
    :param container_path: The path of the container
    :param categories: List of classes to choose, all classes are chosen by default (if empty or omitted)
    :param shuffle: shuffle the list or not
    :param random_state: seed integer to shuffle the dataset
    :return: data and labels of the dataset
    """
    datasets = load_files(container_path=container_path, categories=categories,
                          load_content=load_content, shuffle=shuffle, encoding=encoding,
                          random_state=random_state)
    return datasets


def load_data_labels(anger_data_file, disgust_data_file, fear_data_file, neutral_data_file, sadness_data_file, surprise_data_file):
    """
    Load data and labels
    :param datasets:
    :return:
    """
    anger_examples = list(open(anger_data_file, "r", encoding="UTF8").readlines())
    anger_examples = [s.strip() for s in anger_examples]
    sadness_examples = list(open(sadness_data_file, "r", encoding="UTF8").readlines())
    sadness_examples = [s.strip() for s in sadness_examples]
    surprise_examples = list(open(surprise_data_file, "r", encoding="UTF8").readlines())
    surprise_examples = [s.strip() for s in surprise_examples]
    fear_examples = list(open(fear_data_file, "r", encoding="UTF8").readlines())
    fear_examples = [s.strip() for s in fear_examples]
    disgust_examples = list(open(disgust_data_file, "r", encoding="UTF8").readlines())
    disgust_examples = [s.strip() for s in disgust_examples]
    neutral_examples = list(open(neutral_data_file, "r", encoding="UTF8").readlines())
    neutral_examples = [s.strip() for s in neutral_examples]

    # Split by words
    x_text = anger_examples + disgust_examples + fear_examples + neutral_examples + sadness_examples + surprise_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    anger_labels = [[0, 0, 0, 0, 0, 1] for _ in anger_examples]
    disgust_labels = [[0, 0, 0, 0, 1, 0] for _ in disgust_examples]
    fear_labels = [[0, 0, 0, 1, 0, 0] for _ in fear_examples]
    neutral_labels = [[0, 0, 1, 0, 0, 0] for _ in neutral_examples]
    sadness_labels = [[0, 1, 0, 0, 0, 0] for _ in sadness_examples]
    surprise_labels = [[1, 0, 0, 0, 0, 0] for _ in surprise_examples]
    y = np.concatenate([anger_labels, disgust_labels, fear_labels, neutral_labels, sadness_labels, surprise_labels], 0)
    return [x_text, y]


def load_embedding_vectors_word2vec(vocabulary, filename, binary):
    # load embedding_vectors from the word2vec
    encoding = 'utf-8'
    with open(filename, "rb") as f:
        header = f.readline()
        vocab_size, vector_size = map(int, header.split())
        # initial matrix with random uniform
        embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
        if binary:
            binary_len = np.dtype('float32').itemsize * vector_size
            for line_no in range(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == b' ':
                        break
                    if ch == b'':
                        raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                    if ch != b'\n':
                        word.append(ch)
                word = str(b''.join(word), encoding=encoding, errors='strict')
                idx = vocabulary.get(word)
                if idx != 0:
                    embedding_vectors[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                else:
                    f.seek(binary_len, 1)
        else:
            for line_no in range(vocab_size):
                line = f.readline()
                if line == b'':
                    raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                parts = str(line.rstrip(), encoding=encoding, errors='strict').split(" ")
                if len(parts) != vector_size + 1:
                    raise ValueError("invalid vector on line %s (is this really the text format?)" % (line_no))
                word, vector = parts[0], list(map('float32', parts[1:]))
                idx = vocabulary.get(word)
                if idx != 0:
                    embedding_vectors[idx] = vector
        f.close()
        return embedding_vectors


def load_embedding_vectors_glove(vocabulary, filename, vector_size):
    # load embedding_vectors from the glove
    # initial matrix with random uniform
    embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
    f = open(filename)
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        idx = vocabulary.get(word)
        if idx != 0:
            embedding_vectors[idx] = vector
    f.close()
    return embedding_vectors
