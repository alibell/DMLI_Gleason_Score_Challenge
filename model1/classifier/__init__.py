"""
    Classify images from predictions
"""

from sklearn.ensemble import RandomForestClassifier

isup_rules = {
    0:{
        0:0,
        1:0,
        2:0,
        3:0,
        4:0,
        5:0
    },
    1:{
        0:0,
        1:0,
        2:0,
        3:0,
        4:0,
        5:0
    },
    3:{
        0:1,
        1:1,
        2:1,
        3:1,
        4:2,
        5:4
    },
    4:{
        0:3,
        1:3,
        2:3,
        3:3,
        4:4,
        5:5
    },
    5:{
        0:4,
        1:4,
        2:4,
        3:4,
        4:5,
        5:5
    }
}

class imageClassifier ():
    def __init__ (self):
        self.model = RandomForestClassifier(n_estimators=200)
    
    def preprocessor (self, X):
        X_sum = X.sum(axis=1, keepdims=True)
        X_ = X/X_sum
        
        return X_
        
    def fit (self, X, y):
        X_ = self.preprocessor(X)
        self.model.fit(X_, y)
        
    def predict (self, X):
        X_ = self.preprocessor(X)        
        y_hat = self.model.predict(X_)
        
        return y_hat