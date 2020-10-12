from gensim.models import Word2Vec
from gensim.test.utils import common_texts
import gensim.downloader as api
import pandas as pd
import codecs
import numpy as np

DATA_DIR = 'data/disaster/'
MODEL_SAVE_DIR = 'word2vec/'
EMBEDDING_DIR = 'embeddings/'


def convert_to_binary(embedding_path):
    """
    Here, it takes path to embedding text file provided by glove.
    :param embedding_path: takes path of the embedding which is in text format or any format other than binary.
    :return: a binary file of the given embeddings which takes a lot less time to load.
    """
    f = codecs.open(EMBEDDING_DIR + embedding_path, 'r', encoding='utf-8')
    wv = []
    with codecs.open(EMBEDDING_DIR + embedding_path + ".vocab", "w", encoding='utf-8') as vocab_write:
        count = 0
        for line in f:
            if count == 0:
                pass
            else:
                splitlines = line.split()
                vocab_write.write(splitlines[0].strip())
                vocab_write.write("\n")
                wv.append([float(val) for val in splitlines[1:]])
            count += 1
    np.save(EMBEDDING_DIR + embedding_path + ".npy", np.array(wv))

def load_embeddings_binary(embeddings_path):
    """
    It loads embedding provided by glove which is saved as binary file. Loading of this model is
    about  second faster than that of loading of txt glove file as model.
    :param embeddings_path: path of glove file.
    :return: glove model
    """
    with codecs.open(EMBEDDING_DIR + embeddings_path + '.vocab', 'r', 'utf-8') as f_in:
        index2word = [line.strip() for line in f_in]
    wv = np.load(EMBEDDING_DIR + embeddings_path + '.npy')
    model = {}
    for i, w in enumerate(index2word):
        model[w] = wv[i]
    return model

def get_w2v(sentence, model):
    """
    :param sentence: inputs a single sentences whose word embedding is to be extracted.
    :param model: inputs glove model.
    :return: returns numpy array containing word embedding of all words    in input sentence.
    """
    return np.array([model.get(val, np.zeros(100)) for val in sentence.split()], dtype=np.float64)

df = pd.read_csv('{}train.csv'.format(DATA_DIR))

df.set_index('id', inplace=True)

# print(df['keyword'].value_counts())
# print(df['location'].value_counts())
# print(df.describe(include='O'))

# model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)
# model.save("{}model1.model".format(MODEL_SAVE_DIR))

model = load_embeddings_binary('glove.6B.100d.txt')

