from ai_models.supervised_learning_models.lib.knn_model import KNNModel
from ai_models.supervised_learning_models.types import SupervisedLearningPredictionType
from ai_models.types import ErrorType
import numpy as np
from ai_models.supervised_learning_models.constants.cars_map import CARS_MAP
import graphene

class SupervisedLearningPredictionMutation(graphene.Mutation):
    class Arguments:
        model = graphene.String(required=True)
        input = graphene.List(graphene.Float, required=True)
        

    prediction = graphene.Field(SupervisedLearningPredictionType)
    errors = graphene.Field(ErrorType)
    success = graphene.Boolean()

    def mutate(self, info, model, input):
        if model == 'knn':
            knn_model = KNNModel()
            knn_result = knn_model.predict(np.reshape(input, (1, -1)))
            manufacturer = CARS_MAP[knn_result[0]]
            
            return SupervisedLearningPredictionMutation(success=True, prediction=SupervisedLearningPredictionType(result = {"knn": manufacturer}), errors=None)
        
        return SupervisedLearningPredictionMutation(success=True, prediction=SupervisedLearningPredictionType(result={}, errors=None))
    