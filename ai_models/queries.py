from ai_models.supervised_learning_models.types import SupervisedLearningPredictionType
import graphene



class Query(graphene.ObjectType):
    """ Models query """
    supervised_learning_prediction = graphene.Field(SupervisedLearningPredictionType, model=graphene.String())
    
    """ Models query resolver """
    def resolve_supervised_learning_prediction(self, info, model):
        
        # Nothing here yet.
        return SupervisedLearningPredictionType(result = None)
   