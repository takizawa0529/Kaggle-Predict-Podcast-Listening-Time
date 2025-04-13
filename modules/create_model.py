import os
import sys
import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

from sklearn.cluster import KMeans, DBSCAN, MeanShift, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, Lasso, LassoCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class create_model:
    def __init__(self, df):
        self.path = os.getcwd()  
        self.files = os.listdir(self.path)
        self.df = df
    

    def set_variables(self, exp_cols, target_col):
        self.X = self.df[exp_cols]
        self.y = self.df[target_col]
        return self.X, self.y


    def create_model(self, model_name, params, exp_cols, target_col, train_size):
        if model_name == "RandomForestClassifier":
            self.model = RandomForestClassifier(**params) 
        
        self.X, self.y = self.set_variables(exp_cols, target_col)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, train_size=train_size, random_state=529)
        self.model.fit(self.X_train, self.y_train)

        self.y_pred_train = self.model.predict(self.X_train)
        self.y_pred_test = self.model.predict(self.X_test)

        return self.y_pred_train, self.y_pred_test
    
    
    def evaluate_metrics(self, metrics_switches):
        self.metrics = []
        self.metrics_list = ["Accuracy", "Precision", "Recall", "F1 score", "Confusion Metrix"]
        
        self.metric_functions = {
            'accuracy': accuracy_score,
            'precision': precision_score,
            'recall': recall_score,
            'f1_score': f1_score,
            'confusion_matrix': confusion_matrix
        }
        
        self.results = {
            name: func(self.y_train, self.y_pred_train)
            for name, func, condition in zip(self.metric_functions.keys(), self.metric_functions.values(), metrics_switches)
            if condition
        }

        for key, value in self.results.items():
            print(f"{key}: {value}")

        return self.results

    