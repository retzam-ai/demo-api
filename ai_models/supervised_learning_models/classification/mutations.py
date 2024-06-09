from ai_models.supervised_learning_models.constants.index import DATASETS
from ai_models.supervised_learning_models.lib.model_loader import ModelLoader
from ai_models.supervised_learning_models.types import ClassificationPredictionType
from ai_models.types import ErrorType
import numpy as np
import graphene

class SupervisedLearningPredictionClassificationMutation(graphene.Mutation):
    class Arguments:
        dataset = graphene.String(required=True)
        input = graphene.List(graphene.Float, required=True)
        

    prediction = graphene.Field(ClassificationPredictionType)
    errors = graphene.Field(ErrorType)
    success = graphene.Boolean()

    def mutate(self, info, dataset, input):
        if dataset not in DATASETS:
            return SupervisedLearningPredictionClassificationMutation(success=False, prediction=ClassificationPredictionType(result={}, errors=ErrorType(message="Unsupported dataset")))
            
        try:
            modelLoader = ModelLoader(dataset)
            prediction = modelLoader.predict(np.reshape(input, (1, -1)))
            
            return SupervisedLearningPredictionClassificationMutation(
                success=True, 
                prediction=ClassificationPredictionType(result = prediction), 
                errors=None
            )
        except Exception as e:
            print('Error:', e)
            pass
        
        return SupervisedLearningPredictionClassificationMutation(success=False, prediction=ClassificationPredictionType(result={}, errors=ErrorType(message="Something went wrong")))
    