

import pandas as pd
import featureengineering as fe



## read the  joined csv file
filename = "../../data/train.csv"
data = pd.read_csv(filename, sep=',')


## generate necessary token features for dnews heading and news body
data['header_features'] = data.Headline.apply(lambda x : fe.process(x))
data['content_features'] = data.articleBody.apply(lambda x : fe.process(x))


## generate the similarity features (ordinal)
























