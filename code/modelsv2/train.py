from tqdm import tqdm, tqdm_pandas
import pandas as pd
import string
import numpy as np
import datetime
import os

from functools import reduce

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
## Headline width be 5
      

sentences_articlebody = list(data["content_tokens"])
sentences_headlines = list(data["headline_tokens"])

vocabulary_articlebody = [item for sublist in sentences_articlebody for item in sublist]
vocabulary_headlines = [item for sublist in sentences_headlines for item in sublist]

vocabulary = list(set(vocabulary_headlines + vocabulary_articlebody))

train, test = train_test_split(data, test_size=0.2, random_state=55)



