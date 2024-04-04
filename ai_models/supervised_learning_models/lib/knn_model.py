from joblib import load

class KNNModel:
    
    def __init__(self):
        self.model = load('/ai_models/supervised_learning_models/lib/trained_knn_model.joblib')
        
    def predict(self, X):
        return self.model.predict(X)