import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

''' This script generates the following statistics and figures: 
- Number of concepts for whom their best-predicting rule exceeds some F1 threshold
when used to *directly* predict their *presence* in an image;
- Number of concepts for whom their best-predicting rule exceeds some accuracy threshold
when used to *implicitly* predict their *absence* in an image;
- Histogram of distribution of concepts with respects to the F1 scores of the rules predicting their presence.'''

# Load data:
# These are rules with the highest F1 score for predicting the concepts, constructed
# from labels that individually have the highest F1 score for predicting the concepts.
F1_defs = pd.read_csv('data/all_F1_defs.csv')
num_unique_concepts = len(set(F1_defs['Concept'].tolist()))
raw_data = F1_defs.loc[F1_defs['F1 SCORE'] >= 0] # everything should have F1 score >0, but just in case

data_F1 = [round(x, 4) for x in raw_data['F1 SCORE'].tolist()]
#print(data_F1[:5])
data_implicit_accuracy = raw_data['Implicit Accuracy (tn/(tn+fn))'].tolist()

def reportStats(lst, threshold):
    above_threshold = sum(i > threshold for i in lst)
    ratio = above_threshold/num_unique_concepts
    print("%d out of %d concepts (%f) had an F1 score of at least %f." % 
    (above_threshold, num_unique_concepts, ratio, threshold))

print()
print("Directly predicting the presence of a concept:")
reportStats(data_F1, 0.25)
reportStats(data_F1, 0.5)
print("Implicitly predicting the absence of a concept:")
reportStats(data_implicit_accuracy, 0.9)
reportStats(data_implicit_accuracy, 0.95)
reportStats(data_implicit_accuracy, 0.99)

figs_dir = 'data/figs/'
os.makedirs(figs_dir, exist_ok=True)

def plotHist(filename, data, title, xlabel, ylabel):
    # plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.hist(data, bins=np.linspace(0,0.9, 10))
    plt.axis([0, 1, 0, 900])
    plt.axvline(x=0.25,color='orange',linewidth=3,linestyle='dashed')
    plt.axvline(x=0.5,color='orange',linewidth=3,linestyle='dashed')
    ratio = 0.6
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)

    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.title(title, fontsize=20, y=1.08)
    plt.savefig(filename)
    plt.show()

plotHist(figs_dir+'F1_defs_hist.png', data_F1, 'Distribution of Target Concepts by F1 Scores', 'F1 Score', 'Number of Target Concepts')