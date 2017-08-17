from __future__ import print_function

import sys
sys.path.append('..')

import json
from lib.item_selector import ItemSelector
from lib.model_performance_plotter import plot_learning_curve
import pandas
from pprint import pprint
from sklearn.base import BaseEstimator
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from time import time

"""File to load category mapping from"""
CATEGORY_FILE = '../data/categories.json'
"""File to load data set from"""
DATA_FILE = '../data/201708a.csv'
"""File to save the complete model into"""
MODEL_FILE = '../out/cl_model.pkl'

"""
Load category map to convert from Craigslist categories to our own
local app categories.
"""
with open(CATEGORY_FILE) as handle:
    category_map = json.loads(handle.read())

"""Load example data using Pandas"""
data = pandas.read_csv(DATA_FILE)

# data, _ = train_test_split(data, test_size=0.5)

"""Remove all examples with null fields"""
data = data.dropna()

"""Strip out all "X - by owner", etc. text."""
data['category'], _ = data['category'].str.split(' -', 1).str

"""Remap all Craigslist categories to the categories for our use case"""
data['category'].replace(to_replace=category_map, inplace=True)

"""
Drop all examples with null fields again; this time the categories that
we're skipping.
"""
data = data.dropna()

print(data.category.unique())

"""
GridSearchCV already splits a cross validation data set from the
training set.
"""
train, test = train_test_split(data, test_size=0.1)

"""
Pipeline the process to make it more clear what's going on, use less
memory, and enable faster insertion of new steps.

*FeatureUnion

A FeatureUnion allows for unifying multiple input features so that
the model trains itself on all of them.

*selector
Select this column only for the purposes of this step of the
pipeline.

Example:
    {
        'title': 'Lagavulin 16',
        'description': 'A fine bottle this is.',
        'category': 'Alcohol & Spirits'
    }
    =>
    'Lagavulin 16'

*vect
Embed the words in text using a matrix of token counts.

Example:
    ["dog cat fish", "dog cat", "fish bird", "bird"]
        =>
    [[0 1 1 1],
     [0 2 1 0],
     [1 0 0 1],
     [1 0 0 1]]

*tfidf
Deprioritize words that appear very often, such as
"the", "an", "craigslist", etc.

Example:
    [[3, 0, 1],
     [2, 0, 0],
     [3, 0, 0]]
    =>
    [[ 0.81940995,  0.        ,  0.57320793],
     [ 1.        ,  0.        ,  0.        ],
     [ 1.        ,  0.        ,  0.        ]]
"""
pipeline = Pipeline([
    ('union', FeatureUnion(
        transformer_list=[
            ('title', Pipeline([
                ('selector', ItemSelector(key='title')),
                ('vect', CountVectorizer(stop_words='english',
                                         decode_error='replace',
                                         strip_accents='ascii',
                                         max_df=0.8)),
                ('tfidf', TfidfTransformer(smooth_idf=False))
            ])),
            ('description', Pipeline([
                ('selector', ItemSelector(key='description')),
                ('vect', CountVectorizer(stop_words='english',
                                         decode_error='replace',
                                         strip_accents='ascii',
                                         binary=True,
                                         max_df=0.8,
                                         min_df=10)),
                ('tfidf', TfidfTransformer(smooth_idf=False))
            ]))
        ]
    )),
    ('clf', LogisticRegression(C=5, dual=False, class_weight='balanced'))
])

parameters = {
    # Controls on regression model.
    # 'clf__C': [0.1, 0.3, 1, 3, 5, 10, 30, 100, 300, 1000]
    # 'clf__class_weight': [None, 'balanced'],
    # 'clf__dual': [True, False],

    # Controls on word vectorization.
    # 'union__title__vect__max_df': [0.8, 0.85, 0.9, 0.95, 1],
    # 'union__title__vect__min_df': [1, 10],
    # 'union__title__vect__ngram_range': [(1, 1), (1, 2)],
    # 'union__description__vect__ngram_range': [(1, 1), (1, 2)],
    # 'union__description__vect__max_df': [0.8, 0.85, 0.9, 0.95, 1],
    # 'union__description__vect__min_df': [1, 10, 100],

    # Controls on TfIdf normalization.
    # 'union__title__tfidf__norm': [None, 'l1', 'l2'],
    # 'union__title__tfidf__use_idf': [True, False],
    # 'union__title__tfidf__smooth_idf': [True, False],
    # 'union__title__tfidf__sublinear_tf': [False, True],
    # 'union__description__tfidf__norm': [None, 'l1', 'l2'],
    # 'union__description__tfidf__use_idf': [True, False],
    # 'union__description__tfidf__smooth_idf': [True, False],
    # 'union__description__tfidf__sublinear_tf': [False, True],
}

if __name__ == "__main__":
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=10)

    print('Performing grid search...')
    print('Pipeline: ', [name for name, __ in pipeline.steps])
    print('Parameters: ')
    pprint(parameters)
    t0 = time()
    grid_search.fit(train[['title', 'description']], train['category'])
    print("Done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    score = grid_search.score(test[['title', 'description']], test['category'])
    print("Test accuracy: %f" % score)

    # plot_learning_curve(grid_search.best_estimator_,
    #                     'Item Categorizer',
    #                     train[['title', 'description']],
    #                     train['category'])
    # plt.show()

    joblib.dump(grid_search, MODEL_FILE)
