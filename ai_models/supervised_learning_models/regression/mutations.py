from ai_models.supervised_learning_models.constants.index import DATASETS
from ai_models.supervised_learning_models.lib.model_loader import ModelLoader
from ai_models.supervised_learning_models.types import RegressionPredictionType
from ai_models.types import ErrorType
import numpy as np
import graphene
from joblib import load

class SupervisedLearningPredictionRegressionMutation(graphene.Mutation):
    class Arguments:
        simple = graphene.List(graphene.Float, required=True)
        multiple = graphene.List(graphene.Float, required=True)
        

    prediction = graphene.Field(RegressionPredictionType)
    errors = graphene.Field(ErrorType)
    success = graphene.Boolean()

    def mutate(self, info, simple, multiple):
            
        try:
            model = "/ai_models/supervised_learning_models/trained_models/houses/simple-linear-regression.joblib"
            
            simple_linear = load(model)
            multiple_linear = load(model)
            
            X_simple = np.array(simple).reshape(-1, 1)
            X_multiple = np.array(multiple).reshape(-1, 1)
            simple_linear_prediction = simple_linear.predict(X_simple)
            multiple_linear_prediction = multiple_linear.predict(X_multiple)
            
            
            return SupervisedLearningPredictionRegressionMutation(
                success=True, 
                prediction=RegressionPredictionType(result = {
                    'simple_linear_regression': simple_linear_prediction[0],
                    'multiple_linear_regression': multiple_linear_prediction[0]}), 
                errors=None
            )
        except Exception as e:
            print('Error:', e)
            pass
        
        return SupervisedLearningPredictionRegressionMutation(success=False, prediction=RegressionPredictionType(result={}, errors=ErrorType(message="Something went wrong")))
    