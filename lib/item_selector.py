from sklearn.base import BaseEstimator, TransformerMixin


# See http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html
# #sphx-glr-auto-examples-hetero-feature-union-py
class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]
