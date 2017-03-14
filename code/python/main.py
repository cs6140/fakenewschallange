

import pandas as pd
import featureengineering as fe



## read the  joined csv file
filename = "<to-be-added>"
data = pd.read_csv(filename, sep=',')


## remove stopwords from the article body
data['content'] = data.articleBody.apply(lambda x : fe.tokenize(x))
data['content'] = data.content.apply(lambda x : fe.remove_stopwords(x))






















