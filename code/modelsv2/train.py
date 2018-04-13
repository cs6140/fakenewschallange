from tqdm import tqdm, tqdm_pandas
import pandas as pd
import string
import numpy as np
import datetime
import os

from functools import reduce
import gensim.models.keyedvectors as word2vec
from sklearn.model_selection import train_test_split

tqdm.pandas(tqdm())

__author__ = "Sreejith Sreekumar"
__email__ = "sreekumar.s@husky.neu.edu"
__version__ = "0.0.1"


bodies = "../../data/train_bodies.csv"
stances = "../../data/train_stances.csv"

content = pd.read_csv(bodies, sep=",")
headlines = pd.read_csv(stances, sep=",")


data = pd.merge(content, headlines, how="left", on="Body ID")
data["index"] = data.index


def preprocess(x):
    translator = str.maketrans('','',string.punctuation)
    x = x.replace("“","")
    x = x.replace("”","")
    x = x.replace("‘","")
    x = x.replace("’","")
    return x.lower().translate(translator)

data["content_tokens"] = data["articleBody"].progress_apply(lambda x: preprocess(x))
data["headline_tokens"] = data["Headline"].progress_apply(lambda x: preprocess(x))

data["content_len"] = data["content_tokens"].progress_apply(lambda x : len(x))
data["headline_len"] = data["headline_tokens"].progress_apply(lambda x : len(x))

print("------ Data Statistcs -----")
l = list(data["content_len"])
print("Mean lengths of articles", reduce(lambda x, y: x + y, l) / len(l))
print("Median lengths of articles", np.median(l))

l = list(data["headline_len"])
print("Mean lengths of headlines", reduce(lambda x, y: x + y, l) / len(l))
print("Median lengths of headlines", np.median(l))


## Let's keep the article length 400
## Headline width be 20
      
sentences_articlebody = list(data["content_tokens"])
sentences_headlines = list(data["headline_tokens"])

vocabulary_articlebody = [item for sublist in sentences_articlebody for item in sublist]
vocabulary_headlines = [item for sublist in sentences_headlines for item in sublist]

vocabulary = list(set(vocabulary_headlines + vocabulary_articlebody))

train, test = train_test_split(data, test_size=0.2, random_state=55)




def get_word_vectors(_vocabulary):

    wordslist = list(_vocabulary)
    limit = len(wordslist)

    model_path = "/home/sree/code/dl101/sentiment/amazon-reviews/GoogleNews-vectors-negative300.bin"

    model = word2vec.KeyedVectors.load_word2vec_format(model_path, binary=True)

    print("Word2Vec loaded...")

    invalid_words = []    
    def get_vector(word):
        try:
            return model[word]
        except:
            invalid_words.append(word)
            return limit

    wordvectors = np.zeros([len(wordslist), 300], dtype=np.float32)

    for i, word in tqdm(enumerate(wordslist)):
        wordvectors[i] = get_vector(word)

    del model

    return wordvectors, invalid_words

wordvectors, invalid_words = get_word_vectors(vocabulary)




def get_vectors_of_document(sentence, sequence_len = 400):
    def get_index(word):
        if word in invalid_words:
            return len(vocabulary)
        try:
            return vocabulary.index(word)
        except:
            return len(vocabulary)
    
    words = sentence.split()
    doc_vec = np.zeros(sequence_len)
    sequence =  [get_index(word) for word in words][:sequence_len]
    if(len(sequence) < sequence_len):
        sequence[len(sequence):sequence_len] = [0] * (sequence_len - len(sequence))
    
    return np.asarray(sequence)




data["encoded_article"] = data["content_tokens"].progress_apply(lambda x : get_vectors_of_document(x))
data["encoded_headline"] = data["headline_tokens"].progress_apply(lambda x : get_vectors_of_document(x, 20))

