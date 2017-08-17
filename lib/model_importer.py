# -*- coding: utf-8 -*-

from lib.item_selector import ItemSelector
from sklearn.externals import joblib

MODEL_FILE = 'out/cl_model.pkl'

def model():
    return joblib.load(MODEL_FILE)
