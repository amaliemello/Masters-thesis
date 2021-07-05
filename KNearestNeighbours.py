# The code reads in 'inputML.csv', and saves a figure 'KNearest_lines.png'
# with a plot of different K-values of K-Nearest neighbours against performance score.

import pandas as pd
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import numpy
from sympy import *
import joblib
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import argrelextrema
import numpy as np
from sympy import poly
from sympy.abc import x
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix

col = ['deeppink', 'orangered', 'gold', 'springgreen', 'deepskyblue', 'royalblue']


def calculate_specificity(Y_test_binarized, predicitions_binarized):
    cm1 = confusion_matrix(Y_test_binarized, predicitions_binarized)
    specificity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
    return specificity1


inputML_data = pd.read_csv('inputML.csv', delimiter=';', low_memory=True)
X = inputML_data[1:]
Y = inputML_data[0:1]
Accuracy = []
Sensitivity = []
Specificity = []
Precision = []
F1 = []
Matthews_correlation_coefficient = []
for i in range(1,51):  # 51
    model = KNeighborsClassifier(n_neighbors=i)
    accuracy_sum = []
    recall_sum = []
    specificity_sum = []
    precision_sum = []
    f1_sum = []
    matthews_sum = []
    for i in range(30):  # range 30
        X_train, X_test, Y_train, Y_test = train_test_split(X.transpose(), Y.transpose(), test_size=0.2)
        model.fit(X_train, Y_train)
        predictions = model.predict(X_test)
        #binarizer = MultiLabelBinarizer()
        Y_test2 = (numpy.array(Y_test).transpose())[0]
        Y_test3 = []
        for i in Y_test2:
            Y_test3.append(i)
        predictions3 = []
        for i in predictions:
            predictions3.append(i)

        accuracy = accuracy_score(Y_test3, predictions3)
        recall = recall_score(Y_test3, predictions3, average='macro')
        specificity = calculate_specificity(Y_test3, predictions3)
        precision = precision_score(Y_test3, predictions3, average='macro')
        f1 = f1_score(Y_test3, predictions3, average='macro')
        matthews = matthews_corrcoef(list(Y_test3), list(predictions3))
        accuracy_sum.append(accuracy)
        recall_sum.append(recall)
        specificity_sum.append(specificity)
        precision_sum.append(precision)
        f1_sum.append(f1)
        matthews_sum.append(matthews)

    accuracy_av = sum(accuracy_sum) / len(accuracy_sum)
    recall_av = sum(recall_sum) / len(recall_sum)
    specificity_av = sum(specificity_sum)/ len(specificity_sum)
    precision_av = sum(precision_sum) / len(precision_sum)
    f1_av = sum(f1_sum)/len(f1_sum)
    matthews_av = sum(matthews_sum) / len(matthews_sum)

    Accuracy.append(accuracy_av)
    Sensitivity.append(recall_av)
    Specificity.append(specificity_av)
    Precision.append(precision_av)
    F1.append(f1_av)
    Matthews_correlation_coefficient.append(matthews_av)

predictionScores = [Accuracy, Sensitivity, Specificity, Precision, F1, Matthews_correlation_coefficient]
predictionLabels = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1', 'Matthews correlation coefficient']
fig, axs = plt.subplots(1, 2, sharey=True, sharex=True)
for k in range(len(predictionScores)):
    t = predictionLabels[k]
    range_list = list(range(1, 51))  # 51
    axs[0].plot(range_list, predictionScores[k], color=col[k])  # range_list, Sensitivity, range_list, Specificity, range_list, Precision, range_list, F1)
    mymodel = numpy.poly1d(numpy.polyfit(range_list, predictionScores[k], 3))
    myline = numpy.linspace(1, 50, 50)  # (1,50,50)
    top = sorted(list(mymodel.deriv(1).roots))
    intList = []
    for i in top:
        if i >= 1 and i <= 50:  # 50
            intList.append(i)  # append int(i)?

    #axs[0].plot(intList, mymodel(intList))
    axs[1].plot(myline, mymodel(myline), label="%s" % (t), color=col[k])

    for i in range(len(intList)):
        print("k=%.2f" % intList[i]) #, "%.3f)" % mymodel(intList[i]))
        if i == 0:
            #plt.text(2, (mymodel(intList[i]))-0.015, mymodel)
            print(mymodel)

for ax in axs.flat:
    ax.set(xlabel='k', ylabel='Score')

# Hide redundant labels and ticks.
for ax in axs.flat:
    ax.label_outer()

axs[0].set_title('(a)')
axs[1].set_title('(b)')

# plt.legend(title="Predictor measurement")
plt.legend(title="Performance measure", bbox_to_anchor=(1,-0.2), prop={'size': 6})
fig.tight_layout()

plt.savefig('KNearest_lines.png', dpi=300)
