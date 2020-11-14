from gensim.models import Word2Vec
from gensim.test.utils import common_texts
from cleantext import clean
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
import gensim.downloader as api
import pandas as pd
import codecs
import re
import numpy as np

DATA_DIR = 'data/disaster/'
MODEL_SAVE_DIR = 'word2vec/'
EMBEDDING_DIR = 'embeddings/'
DIMS = 100


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
    :return: returns numpy array containing word embedding of all words in input sentence.
    """
    return np.array([model.get(val, np.zeros(100)) for val in sentence.split()], dtype=np.float64)

def clean_text(text):
    text = clean(text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    return text

''' Load data '''
df = pd.read_csv('{}train.csv'.format(DATA_DIR))
df.set_index('id', inplace=True)

# print(df['keyword'].value_counts())
# print(df['location'].value_counts())
# print(df.describe(include='O'))

# model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)
# model.save("{}model1.model".format(MODEL_SAVE_DIR))

''' Load word vectors '''
model = load_embeddings_binary('glove.6B.100d.txt')

''' Find max tweet length (in words) '''
df['text'] = df['text'].apply(lambda x: clean_text(x))
df['length'] = df['text'].apply(lambda x: len(x.split(' ')))
max_len = max(df['length'])

output_array = np.zeros(shape=(df.shape[0], max_len, DIMS))

for i, txt in enumerate(df['text']):
    tensor = get_w2v(txt, model)
    insert = np.zeros(shape=(max_len, DIMS))
    insert[:tensor.shape[0], :tensor.shape[1]] = tensor
    output_array[i] = insert

X = output_array
y = np.asarray(df['target'])

print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print(X_train.shape)

lstm = Sequential()
lstm.add(LSTM(256, activation='relu', return_sequences=True, input_shape=(max_len, DIMS)))
lstm.add(LSTM(256, activation='relu'))
lstm.add(Dense(1))
lstm.compile(optimizer='adam', loss='mse')
# fit model
lstm.fit(X_train, y_train, epochs=6, verbose=1)

predictions = lstm.predict_classes(X_test)

cm = confusion_matrix(y_test, predictions)

print(cm)