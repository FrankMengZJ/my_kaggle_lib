import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_log_error

class model_detail():
    def __init__(self, name, model, metric, metric_type, ascend = True):
        self.name = name
        self.model = model
        self.metric = metric
        self.metric_type = metric_type
        self.ascend = ascend  
    
    def __lt__(self, other):
        if not isinstance(other, model_detail) or self.metric_type != other.metric_type:
            return NotImplemented
        return self.ascend and (self.metric < other.metric)
    
    def __gt__(self, other):
        if not isinstance(other, model_detail) or self.metric_type != other.metric_type:
            return NotImplemented
        return self.ascend and (self.metric > other.metric)
    
    def __eq__(self, other):
        if not isinstance(other, model_detail) or self.metric_type != other.metric_type:
            return NotImplemented
        return self.ascend and (self.metric == other.metric)


class basic_model_comparison():
    def __init__(self, train, tar_col, metric = r2_score, ascend = True, model_list = set()):
        X = train.drop(columns = tar_col)
        y = train[tar_col]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=1453)
        self.tar_col = tar_col
        self.models = set()
        self.metric = metric
        self.ascend = ascend
        if len(model_list) == 0:
            from sklearn.linear_model import LinearRegression
            from sklearn.tree import DecisionTreeRegressor
            model_list.add(LinearRegression())
            model_list.add(DecisionTreeRegressor(random_state=42))
        else:
            self.model_list = model_list

        for model in model_list:
            self.construct_model(model)
            
    def construct_model(self, model):
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        metric = self.metric(self.y_test, y_pred)
        md = model_detail(type(model), model, metric, self.metric, self.ascend)
        self.models.add(md)

    

    
