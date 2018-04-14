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
    return x.lower().translate(translator).split()

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




def get_vectors_of_document(words, sequence_len = 400):
    def get_index(word):
        if word in invalid_words:
            return len(vocabulary)
        try:
            return vocabulary.index(word)
        except:
            return len(vocabulary)
    
    doc_vec = np.zeros(sequence_len)
    sequence =  [get_index(word) for word in words][:sequence_len]
    if(len(sequence) < sequence_len):
        sequence[len(sequence):sequence_len] = [0] * (sequence_len - len(sequence))
    
    return np.asarray(sequence)





data["encoded_article"] = data["content_tokens"].progress_apply(lambda x : get_vectors_of_document(x))
data["encoded_headline"] = data["headline_tokens"].progress_apply(lambda x : get_vectors_of_document(x, 20))


print("Setting the labels...")
data["label"] = data["Stance"].apply(lambda x: [1, 0, 0, 0] if x == 'agree' else ([0, 1, 0, 0] if x == 'discuss' else ([0, 0, 1, 0] if x == 'disagree' else [0, 0, 0, 1])))


import pickle

with open("data.bin","wb") as f:
    pickle.dump(data, f)

#load vectorized data    
# with open("data.bin","rb") as f:
#     _data = pickle.load(f)

train, test = train_test_split(data, test_size=0.2, random_state=55)

num_classes = 4
batch_size = 500
seq_len_article = 400
seq_len_headline = 20
num_dimensions = 300
input_size = len(data)

from random import randint

def get_train_batch():

    start_index = randint(0, input_size - batch_size)
    end_index = start_index + batch_size
    print("Next batch to train starting index: ", start_index)


    batch_headline = (_data['encoded_headline'][start_index: end_index]).tolist()
    batch_article = (_data['encoded_article'][start_index: end_index]).tolist()
    labels = _data['label'][start_index: end_index].tolist()

    headlines = np.zeros([batch_size, seq_len_headline])
    for i in range(batch_size):
        headlines[i] = batch_headline[i]

    articles = np.zeros([batch_size, seq_len_article])
    for i in range(batch_size):
        articles[i] = batch_article[i]
        
    return headlines, articles, np.array(labels)
    


headlines, articles, labels = get_train_batch()


import tensorflow as tf
tf.reset_default_graph()

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


labels = tf.placeholder(tf.float32, [batch_size, num_classes])

article_input = tf.placeholder(tf.int32, [batch_size, seq_len_article])
article_data = tf.Variable(tf.zeros([batch_size, seq_len_article, num_dimensions]),dtype=tf.float32)
article_data = tf.nn.embedding_lookup(wordvectors, article_input)

headline_input = tf.placeholder(tf.int32, [batch_size, seq_len_headline])
headline_data = tf.Variable(tf.zeros([batch_size, seq_len_headline, num_dimensions]),dtype=tf.float32)
headline_data = tf.nn.embedding_lookup(wordvectors, headline_input)


merged_data = tf.concat([headline_data, article_data], axis=1)

iterations = 10000

for i in range(iterations):
   headlines, articles, labels = get_train_batch();

   sess.run(merged_data, {article_input: articles, headline_input: headlines, labels:labels})
   print("Epoch :", i+1)

   import ipdb
   ipdb.set_trace()
   
   # Write summary to Tensorboard
   if (i % 500 == 0):
       summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})

