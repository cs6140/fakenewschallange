__author__ = 'Ankit'

import seaborn as sns
sns.set_style("whitegrid")
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
    dataFrame.to_excel('test.xlsx', sheet_name='Sheet1')
    print("Excel file generated")

def summaryStatistics(dataFrame):
    print(dataFrame.columns)
    print(dataFrame)
    #print(data.describe())
    print(dataFrame.groupby(['Stance']).mean())

def countPlot(dataFrame):
    fig,(ax1, ax2) = plt.subplots(ncols=2, sharey=False)
    c1 = sns.countplot(x="Stance", data=dataFrame, palette="Greens_d",ax=ax1,order=['unrelated','discuss','agree','disagree'],)
    c1.set(xlabel='Stances in Test Set', ylabel = 'Count')
    c2 = sns.countplot(x="predicted_RF", data=dataFrame, palette="Greens_d",ax = ax2,order=['unrelated','discuss','agree','disagree'])
    c2.set(xlabel='Predicted Stances (Using Random Forest)', ylabel = 'Count')
    plt.show()

def countPlot_headline_article(dataFrame):
    fig,(ax1, ax2) = plt.subplots(ncols=2, sharey=False)
    c1 = sns.countplot(x="char_length_body", data=dataFrame, ax=ax1,color="red")
    c1.set(xlabel='Article Lengths', ylabel = 'Count')
    c2 = sns.countplot(x="char_length_headline", data=dataFrame, ax = ax2, color="red")
    c2.set(xlabel='Headline Lengths', ylabel = 'Count')
    plt.show()

def swarmPlot(dataFrame):
    fig,(ax1, ax2) = plt.subplots(ncols=2, sharey=False)
    c1 = sns.swarmplot(x="Stance", y="Body ID", data=dataFrame,ax=ax1,order=['unrelated','discuss','agree','disagree'])
    c1.set(xlabel='Actual Stances', ylabel = 'Body ID')
    c2 = sns.swarmplot(x="predicted_RF", y="Body ID", data=dataFrame,ax = ax2,order=['unrelated','discuss','agree','disagree'])
    c2.set(xlabel='Predicted Stances (Using Random Forest)', ylabel = 'Body ID')
    plt.show()

def compare_countPlots(dataFrame):
    fig,(ax1, ax2, ax3) = plt.subplots(ncols=3, sharey=False)
    c1 = sns.countplot(x="predicted_XGB", data=dataFrame, hue="Stance",ax=ax1,order=['unrelated','discuss','agree','disagree'],)
    c1.set(xlabel='Predicted Stances (Using XGBoost)', ylabel = 'Count')
    c2 = sns.countplot(x="predicted_RF", data=dataFrame, hue="Stance",ax = ax2,order=['unrelated','discuss','agree','disagree'])
    c2.set(xlabel='Predicted Stances (Using Random Forest)', ylabel = 'Count')
    c3 = sns.countplot(x="predicted_SVM", data=dataFrame, hue="Stance",ax = ax3,order=['unrelated','discuss','agree','disagree'])
    c3.set(xlabel='Predicted Stances (Using SVM)', ylabel = 'Count')
    plt.show()

def pairPlot(dataFrame):
    sns.pairplot(dataFrame,hue='Stance')
    plt.show()


def feature_bodyLength(dataFrame):
    sns.lmplot('char_length_body','overlapping', data = dataFrame, hue='Stance',fit_reg=False)
    plt.show()
    
    sns.lmplot('char_length_body','reoccur1', data = dataFrame, hue='Stance',fit_reg=False)
    plt.show()

    sns.lmplot('char_length_body','reoccur2', data = dataFrame, hue='Stance',fit_reg=False)
    plt.show()

    sns.lmplot('char_length_body','reoccur3', data = dataFrame, hue='Stance',fit_reg=False)
    plt.show()

    sns.lmplot('char_length_body','reoccur4', data = dataFrame, hue='Stance',fit_reg=False)
    plt.show()

    sns.lmplot('char_length_body','reoccur5', data = dataFrame, hue='Stance',fit_reg=False)
    plt.show()

    sns.lmplot('char_length_body','reoccur6', data = dataFrame, hue='Stance',fit_reg=False)
    plt.show()

def pointPlot(dataFrame):
    # plt.show(sns.boxplot(x="Stance", y="cosine", hue="Stance", data=dataFrame))
    # plt.show(sns.boxplot(x="Stance", y="euclidean", hue="Stance", data=dataFrame))
    # plt.show(sns.violinplot(x="Stance", y="cosine", hue="Stance", data=dataFrame))
    # plt.show(sns.violinplot(x="Stance", y="euclidean", hue="Stance", data=dataFrame))
    plt.show(sns.boxplot(x="Stance", y="overlapping", hue="Stance", data=dataFrame))
    # plt.show(sns.violinplot(x="Stance", y="overlapping", hue="Stance", data=dataFrame))

    fig,(ax1, ax2) = plt.subplots(ncols=2, sharey=False)
    c1 = sns.boxplot(x="Stance",y="cosine", data=dataFrame,hue="Stance",ax=ax1,order=['unrelated','discuss','agree','disagree'],)
    c1.set(xlabel='Categorical Stances in the Training Set', ylabel = 'Cosine Distance')
    c2 = sns.boxplot(x="Stance",y="euclidean", data=dataFrame,hue="Stance",ax = ax2,order=['unrelated','discuss','agree','disagree'])
    c2.set(xlabel='Categorical Stances in the Training Set', ylabel = 'Euclidean Distance')
    plt.show()

    fig,(ax1, ax2) = plt.subplots(ncols=2, sharey=False)
    c1 = sns.violinplot(x="Stance",y="cosine", data=dataFrame,hue="Stance",ax=ax1,order=['unrelated','discuss','agree','disagree'],)
    c1.set(xlabel='Categorical Stances in the Training Set', ylabel = 'Cosine Distance')
    c2 = sns.violinplot(x="Stance",y="euclidean", data=dataFrame,hue="Stance",ax = ax2,order=['unrelated','discuss','agree','disagree'])
    c2.set(xlabel='Categorical Stances in the Training Set', ylabel = 'Euclidean Distance')
    plt.show()

    fig,(ax1, ax2) = plt.subplots(ncols=2, sharey=False)
    c1 = sns.boxplot(x="Stance",y="overlapping", data=dataFrame,hue="Stance",ax=ax1,order=['unrelated','discuss','agree','disagree'],)
    c1.set(xlabel='Categorical Stances in the Training Set', ylabel = 'Overlapping Ratio',ylim=(0,0.15))
    c2 = sns.violinplot(x="Stance",y="overlapping", data=dataFrame,hue="Stance",ax = ax2,order=['unrelated','discuss','agree','disagree'])
    c2.set(xlabel='Categorical Stances in the Training Set', ylabel = 'Overlapping Ratio,',ylim=(0.0,0.15))
    plt.show()
    #c1.set(ylim=(0.0,0.4),)






# Also can be check headline counts (would be good to check if more headlines to a topic skew in favor to unrelated articles and fewer towards agree/disagree)


