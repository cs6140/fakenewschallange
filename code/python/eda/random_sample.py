import pandas as pd



## read the  joined csv file
filename = "../../../data/train.csv"
data = pd.read_csv(filename, sep=',')


sample = data.sample(frac=0.05)

sample.to_csv("../../../data/sample.csv", sep=',')


