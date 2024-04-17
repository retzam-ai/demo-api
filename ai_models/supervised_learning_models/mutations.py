from ai_models.supervised_learning_models.constants.index import DATASETS, MODELS
from ai_models.supervised_learning_models.lib.model_loader import ModelLoader
from ai_models.supervised_learning_models.types import SupervisedLearningPredictionType
from ai_models.types import ErrorType
import numpy as np
from ai_models.supervised_learning_models.constants.prediction_maps import CARS_MAP, INJURY_MAP
import graphene

class SupervisedLearningPredictionMutation(graphene.Mutation):
    class Arguments:
        model = graphene.String(required=True)
        dataset = graphene.String(required=True)
        input = graphene.List(graphene.Float, required=True)
        

    prediction = graphene.Field(SupervisedLearningPredictionType)
    errors = graphene.Field(ErrorType)
    success = graphene.Boolean()

    def mutate(self, info, model, dataset, input):
        if model not in MODELS or dataset not in DATASETS:
            return SupervisedLearningPredictionMutation(success=False, prediction=SupervisedLearningPredictionType(result={}, errors=ErrorType(message="Unsupported model or dataset")))
            
        try:
            modelLoader = ModelLoader(dataset)
            result = modelLoader.predict(np.reshape(input, (1, -1)))
            
            if dataset == 'cars':
                prediction = CARS_MAP[result[0]]
            
            if dataset == 'injury':
                prediction = INJURY_MAP[result[0]]
            
            return SupervisedLearningPredictionMutation(success=True, prediction=SupervisedLearningPredictionType(result = {"knn": prediction}), errors=None)
        except:
            pass
        
        return SupervisedLearningPredictionMutation(success=False, prediction=SupervisedLearningPredictionType(result={}, errors=ErrorType(message="Something went wrong")))
    