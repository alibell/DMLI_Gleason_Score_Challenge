"""
    Classify images from predictions
"""

from sklearn.ensemble import RandomForestClassifier

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