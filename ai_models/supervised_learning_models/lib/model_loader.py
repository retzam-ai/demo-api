from joblib import load

class ModelLoader:
    
    def __init__(self, dataset):
        model_name = f"/ai_models/supervised_learning_models/trained_models/{dataset}/{dataset}-knn-model.joblib"
        self.model = load(model_name)
        
    def predict(self, X):
        return self.model.predict(X)