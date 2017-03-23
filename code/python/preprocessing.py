import nltk


stopwords = nltk.corpus.stopwords.words('english')
word_lemmatizer = nltk.WordNetLemmatizer()


def tokenize(line):
    """
    
    Arguments:
    - `line`:
"""
    return line.lower().split()


    
    
def lemmatize(word):
    """
    
    Arguments:
    - `tokens`:
    """
    return word_lemmatizer.lemmatize(word)


    
    
def preprocess(tokens):
    """
    removes stopwords from the array of tokens received
    
    Arguments:
    - `tokens`:
    """
    processed_words = [lemmatize(word) for word in tokens if not word in stopwords]
    return processed_words


    

def generate_ngrams(input_list, n):
    """
    generate n grams

    pass 2 to get bigrams
    pass 3 to get bigrams
    
    Arguments:
    - `tokens`:
    """
    return zip(*[input_list[i:] for i in range(n)])


    

def process(line):
    """
    
    Arguments:
    - `line`:
    """
    tokens = tokenize(line)
    unigrams = preprocess(tokens)

    ## generate bigrams
#    bigrams = list(map(lambda gram : gram[0] + " + " + gram[1], list(generate_ngrams(unigrams, 2))))

    return unigrams