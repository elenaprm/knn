import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def split_input_data(data):
    X = data[['paramA','paramB']]
    y = data['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test


def display_contours(classifier, number_of_neighbors, X_train, y_train):
    plot_decision_regions(X=X_train.values, y=y_train.values, clf=classifier)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Knn with K='+ str(number_of_neighbors), fontweight="bold")
    plt.show()


def knn(nneighbors, X_train, y_train, X_test):
    clf = KNeighborsClassifier(nneighbors)
    clf.fit(X_train, y_train)
    display_contours(clf, nneighbors, X_train, y_train)
    predicted_y = clf.predict(X_test)
    return predicted_y


def evaluateknn(y_predicted, y_test):
    print("Confusion matrix")
    print(confusion_matrix(y_predicted, y_test), "\n")
    print("Classification reports")
    print(classification_report(y_predicted, y_test), "\n")


if __name__ == "__main__":
    input_data = pd.read_csv("A1-inputData.csv")
    X_train, X_test, y_train, y_test = split_input_data(input_data)
    nneighbors = 3
    predicted_y = knn(nneighbors, X_train, y_train, X_test)
    print("predicted_y\n", predicted_y, "\n")
    evaluateknn(predicted_y, y_test)
