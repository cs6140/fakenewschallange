import nltk
stopwords = nltk.corpus.stopwords.words('english')


def tokenize(line):
    """
    
    Arguments:
    - `line`:
    """
    return line.lower().split()

    


def remove_stopwords(tokens):
    """
    removes stopwords from the array of tokens received
    
    Arguments:
    - `tokens`:
    """
    words = [word for word in tokens if not word in stopwords and word.isalpha()]
    return words


    












