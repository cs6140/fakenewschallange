__author__ = 'Ankit'

import seaborn as sns
import matplotlib.pyplot as plt

# Plot the overlapping ratio with respect to Stance
def plot_overlapping(dataFrame):
    #sns.set(style="whitegrid")
    sns.lmplot('Body ID','overlapping', data = dataFrame, hue='Stance',fit_reg=False)
    plt.show()

 # Plot the character length headline
def plot_headlineLength(dataFrame):
    dataFrame['char_length_headline'].plot()
    plt.show()

def plot_HLS(dataFrame):
    sns.lmplot('Body ID','char_length_headline', data = dataFrame, hue='Stance',fit_reg=False)
    plt.show()

def plot_CLS(dataFrame):
    sns.lmplot('Body ID','char_length_body', data = dataFrame, hue='Stance',fit_reg=False)
    plt.show()

def dataFrame_CSV(dataFrame):
    dataFrame.to_excel('dataframe.xlsx', sheet_name='Sheet1')
    print("Excel file generated")

def summaryStatistics(dataFrame):
    print(dataFrame.columns)
    print(dataFrame)
    #print(data.describe())
    print(dataFrame.groupby(['Stance']).mean())



# Also can be check headline counts (would be good to check if more headlines to a topic skew in favor to unrelated articles and fewer towards agree/disagree)


