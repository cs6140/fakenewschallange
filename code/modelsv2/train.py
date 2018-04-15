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

# print("------ Data Statistcs -----")
# l = list(data["content_len"])
# print("Mean lengths of articles", reduce(lambda x, y: x + y, l) / len(l))
# print("Median lengths of articles", np.median(l))

# l = list(data["headline_len"])
# print("Mean lengths of headlines", reduce(lambda x, y: x + y, l) / len(l))
# print("Median lengths of headlines", np.median(l))


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


# def get_vectors_of_document(words, sequence_len = 400):
#     def get_index(word):
#         if word in invalid_words:
#             return len(vocabulary)
#         try:
#             return vocabulary.index(word)
#         except:
#             return len(vocabulary)
    
#     doc_vec = np.zeros(sequence_len)
#     sequence =  [get_index(word) for word in words][:sequence_len]
#     if(len(sequence) < sequence_len):
#         sequence[len(sequence):sequence_len] = [0] * (sequence_len - len(sequence))
    
#     return np.asarray(sequence)



# data["encoded_article"] = data["content_tokens"].progress_apply(lambda x : get_vectors_of_document(x))
# data["encoded_headline"] = data["headline_tokens"].progress_apply(lambda x : get_vectors_of_document(x, 20))


print("Setting the labels...")
data["label"] = data["Stance"].apply(lambda x: [1, 0, 0, 0] if x == 'agree' else ([0, 1, 0, 0] if x == 'discuss' else ([0, 0, 1, 0] if x == 'disagree' else [0, 0, 0, 1])))


import pickle

# with open("data.bin","wb") as f:
#     pickle.dump(data, f)

#load vectorized data    
with open("data.bin","rb") as f:
    data = pickle.load(f)

train, test = train_test_split(data, test_size=0.2, random_state=55)

num_classes = 4
batch_size = 500
seq_len_article = 400
seq_len_headline = 20
num_dimensions = 300
input_size = len(train)
input_size_test = len(test)

from random import randint

def get_train_batch():

    start_index = randint(0, input_size - batch_size)
    end_index = start_index + batch_size
    print("Next batch to train starting index: ", start_index)


    batch_headline = (train['encoded_headline'][start_index: end_index]).tolist()
    batch_article = (train['encoded_article'][start_index: end_index]).tolist()
    labels = train['label'][start_index: end_index].tolist()

    headlines = np.zeros([batch_size, seq_len_headline])
    for i in range(batch_size):
        headlines[i] = batch_headline[i]

    articles = np.zeros([batch_size, seq_len_article])
    for i in range(batch_size):
        articles[i] = batch_article[i]
        
    return headlines, articles, labels
    


def get_test_batch():

    start_index = randint(0, input_size_test - batch_size)
    end_index = start_index + batch_size
    print("Next batch to train starting index: ", start_index)


    batch_headline = (test['encoded_headline'][start_index: end_index]).tolist()
    batch_article = (test['encoded_article'][start_index: end_index]).tolist()
    labels = test['label'][start_index: end_index].tolist()

    headlines = np.zeros([batch_size, seq_len_headline])
    for i in range(batch_size):
        headlines[i] = batch_headline[i]

    articles = np.zeros([batch_size, seq_len_article])
    for i in range(batch_size):
        articles[i] = batch_article[i]
        
    return headlines, articles, labels
    


headlines, articles, labels = get_train_batch()


import tensorflow as tf
tf.reset_default_graph()


lstmunits = 128

labels = tf.placeholder(tf.float32, [batch_size, num_classes])

article_input = tf.placeholder(tf.int32, [batch_size, seq_len_article])
article_data = tf.Variable(tf.zeros([batch_size, seq_len_article, num_dimensions]),dtype=tf.float32)
article_data = tf.nn.embedding_lookup(wordvectors, article_input)

headline_input = tf.placeholder(tf.int32, [batch_size, seq_len_headline])
headline_data = tf.Variable(tf.zeros([batch_size, seq_len_headline, num_dimensions]),dtype=tf.float32)
headline_data = tf.nn.embedding_lookup(wordvectors, headline_input)


merged_data = tf.concat([headline_data, article_data], axis=1)

lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstmunits)
lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=0.75)
value, _ = tf.nn.dynamic_rnn(lstm_cell, merged_data, dtype=tf.float32)

weight = tf.Variable(tf.truncated_normal([lstmunits, num_classes]), dtype=tf.float32)
bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)


correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

_accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)

tf.summary.scalar('Loss', loss)
tf.summary.scalar('Train Accuracy', accuracy)
tf.summary.scalar('Test Accuracy', _accuracy)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"

sess = tf.InteractiveSession()
writer = tf.summary.FileWriter(logdir, sess.graph)
saver = tf.train.Saver()


iterations = 10000

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


for i in range(iterations):
   headlines, articles, _labels = get_train_batch();

   sess.run(optimizer, {article_input: articles, headline_input: headlines, labels: _labels})
   print("Epoch :", i+1)

   if (i % 50 == 0):

       headlines, articles, _labels = get_test_batch();
       sess.run(_accuracy, {article_input: articles, headline_input: headlines, labels: _labels})

       summary = sess.run(merged, {article_input: articles, headline_input: headlines, labels: _labels})
       
       writer.add_summary(summary, i)

   #Save the network every 10,000 training iterations
   if (i % 1000 == 0 and i != 0):
       save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
       print("saved to %s" % save_path)

       
writer.close()

