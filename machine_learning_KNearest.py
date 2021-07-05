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



def runModelAccuracy(X, Y, model, modelName, i):
    # print(f'\rRunning {modelName}, run {i}', end='')
    X_train, X_test, Y_train, Y_test = train_test_split(
        X.transpose(), Y.transpose(), test_size=0.2)
    model.fit(X_train, np.ravel(Y_train))
    predictions = model.predict(X_test)
    Y_test_list = list((np.array(Y_test).transpose())[0])
    predictions_list = list(predictions)
    Y_test_binarized = Y_test_list  # = binarizer.transform(Y_test_list)
    predictions_binarized = predictions_list  # = binarizer.transform(predictions_list)
    scores = [
        accuracy_score(Y_test_binarized, predictions_binarized)
    ]
    data = {
        'k': [modelName] ,
        'scoringAlgorithm': ['accuracy'],#, 'sensitivity', 'specificity', 'precision', 'f1', 'matthews_corrcoef'],
        'Score': scores,
        'run': [i]
    }
    df = pd.DataFrame(data)
    return df

def runModelSensitivity(X, Y, model, modelName, i):
    # print(f'\rRunning {modelName}, run {i}', end='')
    X_train, X_test, Y_train, Y_test = train_test_split(
        X.transpose(), Y.transpose(), test_size=0.2)
    model.fit(X_train, np.ravel(Y_train))
    predictions = model.predict(X_test)
    Y_test_list = list((np.array(Y_test).transpose())[0])
    predictions_list = list(predictions)
    Y_test_binarized = Y_test_list  # = binarizer.transform(Y_test_list)
    predictions_binarized = predictions_list  # = binarizer.transform(predictions_list)
    scores = [
        recall_score(Y_test_binarized, predictions_binarized, pos_label=1, average='macro')
    ]
    data = {
        'k': [modelName] ,
        'scoringAlgorithm': ['sensitivity'],  # 'specificity', 'precision', 'f1', 'matthews_corrcoef'],
        'Score': scores,
        'run': [i]
    }
    df = pd.DataFrame(data)
    return df

def runModelSpecificity(X, Y, model, modelName, i):
    # print(f'\rRunning {modelName}, run {i}', end='')
    X_train, X_test, Y_train, Y_test = train_test_split(
        X.transpose(), Y.transpose(), test_size=0.2)
    model.fit(X_train, np.ravel(Y_train))
    predictions = model.predict(X_test)
    Y_test_list = list((np.array(Y_test).transpose())[0])
    predictions_list = list(predictions)
    Y_test_binarized = Y_test_list  # = binarizer.transform(Y_test_list)
    predictions_binarized = predictions_list  # = binarizer.transform(predictions_list)
    scores = [
        calculate_specificity(Y_test_list, predictions_list)
    ]
    data = {
        'k': [modelName] ,
        'scoringAlgorithm': ['specificity'],
        'Score': scores,
        'run': [i]
    }
    df = pd.DataFrame(data)
    return df


def runModelPrecision(X, Y, model, modelName, i):
    # print(f'\rRunning {modelName}, run {i}', end='')
    X_train, X_test, Y_train, Y_test = train_test_split(
        X.transpose(), Y.transpose(), test_size=0.2)
    model.fit(X_train, np.ravel(Y_train))
    predictions = model.predict(X_test)
    Y_test_list = list((np.array(Y_test).transpose())[0])
    predictions_list = list(predictions)
    Y_test_binarized = Y_test_list  # = binarizer.transform(Y_test_list)
    predictions_binarized = predictions_list  # = binarizer.transform(predictions_list)
    scores = [
        precision_score(Y_test_binarized, predictions_binarized, average='macro')
    ]
    data = {
        'k': [modelName],
        'scoringAlgorithm': ['precision'],
        'Score': scores,
        'run': [i]
    }
    df = pd.DataFrame(data)
    return df

def runModelF1(X, Y, model, modelName, i):
    # print(f'\rRunning {modelName}, run {i}', end='')
    X_train, X_test, Y_train, Y_test = train_test_split(
        X.transpose(), Y.transpose(), test_size=0.2)
    model.fit(X_train, np.ravel(Y_train))
    predictions = model.predict(X_test)
    Y_test_list = list((np.array(Y_test).transpose())[0])
    predictions_list = list(predictions)
    Y_test_binarized = Y_test_list  # = binarizer.transform(Y_test_list)
    predictions_binarized = predictions_list  # = binarizer.transform(predictions_list)
    scores = [
        f1_score(Y_test_binarized, predictions_binarized, average='macro')
    ]
    data = {
        'k': [modelName],
        'scoringAlgorithm': ['f1'],
        'Score': scores,
        'run': [i]
    }
    df = pd.DataFrame(data)
    return df

def runModelMatthews_corrcoef(X, Y, model, modelName, i):
    # print(f'\rRunning {modelName}, run {i}', end='')
    X_train, X_test, Y_train, Y_test = train_test_split(
        X.transpose(), Y.transpose(), test_size=0.2)
    model.fit(X_train, np.ravel(Y_train))
    predictions = model.predict(X_test)
    Y_test_list = list((np.array(Y_test).transpose())[0])
    predictions_list = list(predictions)
    Y_test_binarized = Y_test_list  # = binarizer.transform(Y_test_list)
    predictions_binarized = predictions_list  # = binarizer.transform(predictions_list)
    scores = [
        matthews_corrcoef(Y_test_list, predictions_list)
    ]
    data = {
        'k': [modelName],
        'scoringAlgorithm': ['matthews_corrcoef'],
        'Score': scores,
        'run': [i]
    }
    df = pd.DataFrame(data)
    return df


def runAllModelsAccuracy(n=1):
    startTime = time.time()
    models = {}
    keys = range(1, 31)
    values = []
    for i in range(1, 31):
        values.append(KNeighborsClassifier(n_neighbors=(i)))
    for i in keys:
        models[i] = values[(i - 1)]

    X, Y = readData()

    columns = ['k', 'scoringAlgorithm', 'Score', 'run']
    df = pd.DataFrame(data=None, columns=columns)

    for name, model in models.items():
        modelTime = time.time()
        for i in range(n):
            result = runModelAccuracy(X, Y, model, name, i + 1)  # saveModels)
            df = df.append(result, ignore_index=True)
            # print(f'Finished running {name} in {(time.time()-modelTime)/60:.1f}')
    # print(df)
    df.to_csv('results_{n}_simulations_KNearest_accuracy.csv')
    # print(f'Finished {n} simulations in {(time.time()-startTime)/60:.1f} minutes.\n')
    return df


def runAllModelsSensitivity(n=1):
    startTime = time.time()
    models = {}
    keys = range(1, 31)
    values = []
    for i in range(1, 31):
        values.append(KNeighborsClassifier(n_neighbors=(i)))
    for i in keys:
        models[i] = values[(i - 1)]

    X, Y = readData()

    columns = ['k', 'scoringAlgorithm', 'Score', 'run']
    df = pd.DataFrame(data=None, columns=columns)

    for name, model in models.items():
        modelTime = time.time()
        for i in range(n):
            result = runModelSensitivity(X, Y, model, name, i + 1)  # saveModels)
            df = df.append(result, ignore_index=True)
            # print(f'Finished running {name} in {(time.time()-modelTime)/60:.1f}')
    # print(df)
    df.to_csv('results_{n}_simulations_KNearest_sensitivity.csv')
    # print(f'Finished {n} simulations in {(time.time()-startTime)/60:.1f} minutes.\n')
    return df

def runAllModelsSpecificity(n=1):
    startTime = time.time()
    models = {}
    keys = range(1, 31)
    values = []
    for i in range(1, 31):
        values.append(KNeighborsClassifier(n_neighbors=i))
    for i in keys:
        models[i] = values[(i - 1)]

    X, Y = readData()

    columns = ['k', 'scoringAlgorithm', 'Score', 'run']
    df = pd.DataFrame(data=None, columns=columns)

    for name, model in models.items():
        modelTime = time.time()
        for i in range(n):
            result = runModelSpecificity(X, Y, model, name, i + 1)  # saveModels)
            df = df.append(result, ignore_index=True)
            # print(f'Finished running {name} in {(time.time()-modelTime)/60:.1f}')
    # print(df)
    df.to_csv('results_{n}_simulations_KNearest_specificity.csv')
    # print(f'Finished {n} simulations in {(time.time()-startTime)/60:.1f} minutes.\n')
    return df

def runAllModelsPrecision(n=1):
    startTime = time.time()
    models = {}
    keys = range(1, 31)
    values = []
    for i in range(1, 31):
        values.append(KNeighborsClassifier(n_neighbors=i))
    for i in keys:
        models[i] = values[(i - 1)]

    X, Y = readData()

    columns = ['k', 'scoringAlgorithm', 'Score', 'run']
    df = pd.DataFrame(data=None, columns=columns)

    for name, model in models.items():
        modelTime = time.time()
        for i in range(n):
            result = runModelPrecision(X, Y, model, name, i + 1)  # saveModels)
            df = df.append(result, ignore_index=True)
            # print(f'Finished running {name} in {(time.time()-modelTime)/60:.1f}')
    # print(df)
    df.to_csv('results_{n}_simulations_KNearest_precision.csv')
    # print(f'Finished {n} simulations in {(time.time()-startTime)/60:.1f} minutes.\n')
    return df

def runAllModelsF1(n=1):
    startTime = time.time()
    models = {}
    keys = range(1, 31)
    values = []
    for i in range(1, 31):
        values.append(KNeighborsClassifier(n_neighbors=(i)))
    for i in keys:
        models[i] = values[(i - 1)]

    X, Y = readData()

    columns = ['k', 'scoringAlgorithm', 'Score', 'run']
    df = pd.DataFrame(data=None, columns=columns)

    for name, model in models.items():
        modelTime = time.time()
        for i in range(n):
            result = runModelF1(X, Y, model, name, i + 1)  # saveModels)
            df = df.append(result, ignore_index=True)
            # print(f'Finished running {name} in {(time.time()-modelTime)/60:.1f}')
    # print(df)
    df.to_csv('results_{n}_simulations_KNearest_f1.csv')
    # print(f'Finished {n} simulations in {(time.time()-startTime)/60:.1f} minutes.\n')
    return df

def runAllModelsMatthews_corrcoef(n=1):
    startTime = time.time()
    models = {}
    keys = range(1, 31)
    values = []
    for i in range(1, 31):
        values.append(KNeighborsClassifier(n_neighbors=i))
    for i in keys:
        models[i] = values[(i - 1)]

    X, Y = readData()

    columns = ['k', 'scoringAlgorithm', 'Score', 'run']
    df = pd.DataFrame(data=None, columns=columns)

    for name, model in models.items():
        modelTime = time.time()
        for i in range(n):
            result = runModelMatthews_corrcoef(X, Y, model, name, i + 1)  # saveModels)
            df = df.append(result, ignore_index=True)
            # print(f'Finished running {name} in {(time.time()-modelTime)/60:.1f}')
    # print(df)
    df.to_csv('results_{n}_simulations_KNearest_matthews_corrcoef.csv')
    # print(f'Finished {n} simulations in {(time.time()-startTime)/60:.1f} minutes.\n')
    return df


def plotResults(results,j,k, col):
    #fig, ax = plt.subplots()
    #fig.set_size_inches(10, 6)

    sns.boxplot(x="k", y="Score",
                     data=df, width=0.6, ax=axs[j,k], color=col)  # removed hue?hue="scoringAlgorithm"
    #plt.savefig('boxplot_KNearest_accuracy.png', dpi=300)


if __name__ == '__main__':
    #df = pd.read_csv('results_{n}_simulations_KNearest.csv')
    fig, axs = plt.subplots(3, 2, sharey=True, sharex=True)
    fig.set_size_inches(11, 6)
    df = runAllModelsAccuracy(n=30)
    plotResults(df,0,0, 'deeppink')
    axs[0,0].set_title("Accuracy")
    df = runAllModelsSensitivity(n=30)
    plotResults(df,0,1, 'orangered')
    axs[0,1].set_title("Sensitivity")
    df = runAllModelsSpecificity(n=30)
    plotResults(df,1,0, 'gold')
    axs[1,0].set_title("Specificity")
    df = runAllModelsPrecision(n=30)
    plotResults(df,1,1, 'springgreen')
    axs[1,1].set_title("Precision")
    df = runAllModelsF1(n=30)
    plotResults(df,2,0,'deepskyblue' )
    axs[2,0].set_title("F1")
    df = runAllModelsMatthews_corrcoef(n=30)
    plotResults(df,2,1, 'royalblue')
    axs[2,1].set_title("Matthews correlation coefficient")

    plt.xticks(ticks=[4, 9, 14, 19, 24, 29])
    # Hide labels and tick labels.
    for ax in axs.flat:
        ax.label_outer()
    #fig.tight_layout()
    #plt.title("Prediction scores for the classifier implementing the k-nearest neighbors vote")
    plt.savefig('boxplot_KNearest_all.png', dpi=300)
