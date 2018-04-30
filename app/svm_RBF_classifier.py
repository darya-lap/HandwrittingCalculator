# Standard scientific Python imports
import time
import datetime as dt

# Import datasets, classifiers and performance metrics
import scipy
from sklearn import datasets, svm, metrics
# fetch original mnist dataset
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# Standard scientific Python imports
from matplotlib.colors import Normalize
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import csv
from http.server import HTTPServer, CGIHTTPRequestHandler


def main():
    server_address = ("", 8000)
    httpd = HTTPServer(server_address, CGIHTTPRequestHandler)
    httpd.serve_forever()

    all_variables = scipy.io.loadmat('data_MNIST.mat')
    X_test = np.array(all_variables['Xtest'])
    X_train = np.array(all_variables['Xtrain'])
    y_test = np.array(all_variables['ytest'])
    y_train = np.array(all_variables['ytrain'])

    gamma_range = np.outer(np.logspace(-3, 0, 4), np.array([1, 5]))
    gamma_range = gamma_range.flatten()

    # generate matrix with all C
    C_range = np.outer(np.logspace(-1, 1, 3), np.array([1, 5]))
    # flatten matrix, change to 1D numpy array
    C_range = C_range.flatten()

    parameters = {'kernel': ['rbf'], 'C': C_range, 'gamma': gamma_range}

    svm_clsf = svm.SVC()
    grid_clsf = GridSearchCV(estimator=svm_clsf, param_grid=parameters, n_jobs=1, verbose=2)

    print('Время начала поиска параметров {}'.format(str(dt.datetime.now())))

    grid_clsf.fit(X_train, y_train)

    print('Время окончания поиска параметров {}'.format(str(dt.datetime.now())))
    sorted(grid_clsf.cv_results_.keys())

    classifier = grid_clsf.best_estimator_
    params = grid_clsf.best_params_

    scores = grid_clsf.cv_results_['mean_test_score'].reshape(len(C_range),
                                                              len(gamma_range))
    # Now predict the value of the test
    expected = y_test
    print('Время начала классификации 10000 символов {}'.format(str(dt.datetime.now())))
    predicted = classifier.predict(X_test)
    print('Время окончания классификации 10000 символов {}'.format(str(dt.datetime.now())))

    print("Отчет для классификатора %s:\n%s\n"
          % (classifier, metrics.classification_report(expected, predicted)))

    cm = metrics.confusion_matrix(expected, predicted)
    print("Точность={}".format(metrics.accuracy_score(expected, predicted)))



class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


if __name__ == "__main__":  # If run as a script, create a test object
    main()
