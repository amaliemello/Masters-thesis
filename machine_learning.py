# The code reads in 'inputML.csv' and saves ML models and 'boxplot.png' with performance scores of the ML algorithms.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import pickle
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


def readData(filename='inputML.csv'):
    inputML_data = pd.read_csv(filename, delimiter=';', low_memory=False)
    X = inputML_data[1:]
    Y = inputML_data[0:1]
    Y_NY = ((pd.DataFrame(Y)).transpose())
    Y_array = np.array(Y_NY)
    Y_binary = []
    for i in Y_array:
        if i == 'Diabetes':
            Y_binary.append('1')
        else:
            Y_binary.append('0')
    Y = pd.DataFrame(Y_binary)
    Y = Y.transpose()
    return X, Y


def calculate_specificity(Y_test_binarized, predictions_binarized):
    cm1 = confusion_matrix(Y_test_binarized, predictions_binarized)
    specificity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
    return specificity1


def runModel(X, Y, model, modelName, i, saveModels=True):
    # print(f'\rRunning {modelName}, run {i}', end='')

    X_train, X_test, Y_train, Y_test = train_test_split(
        X.transpose(), Y.transpose(), test_size=0.2)

    model.fit(X_train, np.ravel(Y_train)) 
    predictions = model.predict(X_test)

    Y_test_list = list((np.array(Y_test).transpose())[0])
    predictions_list = list(predictions)

    #binarizer = MultiLabelBinarizer()

    #binarizer.fit(Y_test_list)
    
    Y_test_binarized = Y_test_list  # = binarizer.transform(Y_test_list)

    predictions_binarized = predictions_list  # = binarizer.transform(predictions_list)

    scores = [
        accuracy_score(Y_test_binarized, predictions_binarized),
        recall_score(Y_test_binarized, predictions_binarized, pos_label=1, average='macro'),
        calculate_specificity(Y_test_list, predictions_list),
        precision_score(Y_test_binarized, predictions_binarized, average='macro'),
        f1_score(Y_test_binarized, predictions_binarized, average='macro'),
        matthews_corrcoef(Y_test_list, predictions_list)
    ]
    data = {
        'Algorithm': [modelName]*6,
        'Performance measure':['Accuracy','Sensitivity', 'Specificity', 'Precision','F1', 'Matthews correlation coefficient'],
        'Score': scores,
        'run': [i]*6
    }
    df = pd.DataFrame(data)

    if saveModels:
        if modelName == 'Logistic regression':
            joblib.dump(model, 'models/model_LogisticRegression.joblib')
        elif modelName == 'Decision tree':
            joblib.dump(model, 'models/model_DecisionTree.joblib')
        elif modelName == 'K-nearest neighbours':
            joblib.dump(model, 'models/model_KNearestNeighbours.joblib')
        elif modelName == 'Random forest':
            joblib.dump(model, 'models/model_RandomForest.joblib')
        elif modelName == 'MLP':
            joblib.dump(model, 'models/model_MLP.joblib')
        else:
            print(modelName)
    
        # Saving decision tree to file
        if modelName == 'Decision tree':
            export_graphviz(model, out_file='dot_files/DecisionTree-model{i}.dot',
                            class_names=['Diabetes', 'Not diabetes'])

    return df


def runAllModels(n=1, saveModels=True):
    startTime = time.time()
    models = {
        'Logistic regression': LogisticRegression(penalty='l2'),
        'Decision tree': DecisionTreeClassifier(),
        'K-nearest neighbours': KNeighborsClassifier(n_neighbors=15),
        'Random forest': RandomForestClassifier(),
        'MLP': MLPClassifier()
    }
    # Removed 'LogisticRegressionl1': LogisticRegression(penalty='l1'),
    #ValueError: Solver lbfgs supports only 'l2' or 'none' penalties, got l1 penalty.
    X, Y = readData()

    columns = ['Algorithm', 'Performance measure', 'Score','run']
    df = pd.DataFrame(data=None, columns=columns)
    
    for name, model in models.items():
        modelTime = time.time()
        for i in range(n):
            result = runModel(X, Y, model, name, i+1, saveModels)
            df = df.append(result, ignore_index=True)
            # print(f'Finished running {name} in {(time.time()-modelTime)/60:.1f}')
    
    # print(df)
    df.to_csv('results_{n}_simulations.csv')
    
    # print(f'Finished {n} simulations in {(time.time()-startTime)/60:.1f} minutes.\n')
    return df


def plotResults(results):
    col=['deeppink', 'orangered', 'gold','springgreen', 'deepskyblue', 'royalblue']
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 6)
    
    ax = sns.boxplot(x="Algorithm", y="Score", hue="Performance measure", data=df, palette=col, width=0.6)
    
    plt.savefig('boxplot.png', dpi=300)
    
    
if __name__ == '__main__':
    df = pd.read_csv('results.csv')
    df = runAllModels(n = 30)
    plotResults(df)
