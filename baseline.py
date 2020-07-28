from sklearn import base

from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

import dill
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from dataloader import IMDBDataSet
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

class Baseline():
    def __init__(self):
        super().__init__()

if __name__ == '__main__':
    imdb = IMDBDataSet()
    text, labels = imdb.reviews(ret_val=True)

    train_dt, test_dt, train_lb, test_lb = train_test_split(text, labels, test_size=0.2, random_state=42)
    bag_of_words_est = Pipeline([('cvt', CountVectorizer(min_df=0.005)), ('SGD', SGDClassifier())])

    bag_of_words_est.fit(train_dt, train_lb)
    dill.dump(bag_of_words_est, open('bag_of_words_est', 'wb'))

    y_pred = bag_of_words_est.predict(test_dt)

    # error = mean_squared_error(y_pred, test_lb)
    #
    # print(error)

