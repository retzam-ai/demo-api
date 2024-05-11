from joblib import load

from ai_models.supervised_learning_models.constants.index import MODELS, MODELS_SLUGS
from ai_models.supervised_learning_models.constants.prediction_maps import CARS_MAP, INJURY_MAP, MACHINES_MAP, YES_NO_MAP

class ModelLoader:
    
    def __init__(self, dataset):
        self.dataset = dataset
        
        # Load models for the given dataset into a list of dictionaries
        self.models = []
        for model in MODELS:
            model_name = f"/ai_models/supervised_learning_models/trained_models/{dataset}/{dataset}-{model}-model.joblib"
            item = {
                'model': load(model_name),
                'name': MODELS_SLUGS[model],
            }
            self.models.append(item)
        
    def predict(self, X):
        predictions = {}
        for model in self.models:
            prediction = model['model'].predict(X)
            
            # Format prediction
            if self.dataset == 'cars':
                predictions[model['name']] = CARS_MAP[prediction[0]]
                
            elif self.dataset == 'injury':
                predictions[model['name']] = INJURY_MAP[prediction[0]]
            
            elif self.dataset == 'machines':
                predictions[model['name']] = MACHINES_MAP[prediction[0]]
                
            elif self.dataset == 'marketing' or self.dataset == 'diabetes':
                predictions[model['name']] = YES_NO_MAP[prediction[0]]
                
                
        return predictions