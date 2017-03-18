
from gensim.models.doc2vec import LabeledSentence
from gensim.models import doc2vec


import pandas as pd
import featureengineering as fe


## Reference : http://stackoverflow.com/questions/22129943/how-to-calculate-the-sentence-similarity-using-word2vec-model-of-gensim-with-pyt


## https://www.insight.io/github.com/piskvorky/gensim/blob/HEAD/gensim/models/doc2vec.py

filename = "../../data/train.csv"
data = pd.read_csv(filename, sep=',')

uid = -1

def make_sentences(words, label = False):
    """
    
    Arguments:
    - `words`:
    - `label`:
    """
    global uid
    uid += 1
    return LabeledSentence(words=words, tags=[uid])



data['header_features'] = data.Headline.apply(lambda x : fe.process(x))
data['content_features'] = data.articleBody.apply(lambda x : fe.process(x))

data['all_words'] = data.header_features + data.content_features

data['vectorization_input']  = data.all_words.apply(lambda x : make_sentences(x))

#data['labelled'] = data[['all_words','Stance']].apply(lambda x: make_sentences(*x), axis=1)


# data['labelled_sentence'] = data.all_words.apply(lambda x : make_sentences)

sentences = data.vectorization_input.tolist()

model = doc2vec.Doc2Vec(sentences, size = 300, window = 300, min_count = 1, workers = 4)


# model = Doc2Vec(alpha=0.025, min_alpha=0.025)  # use fixed learning rate
# model.build_vocab(sentences)
# for epoch in range(10):
#     model.train(sentences)
#     model.alpha -= 0.002  # decrease the learning rate
#     model.min_alpha = model.alpha      




#http://stackoverflow.com/questions/29760935/how-to-get-vector-for-a-sentence-from-the-word2vec-of-tokens-in-sentence

# https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-lee.ipynb

model.infer_vector(["this","is", "an", "example"])
