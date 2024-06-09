from ai_models.supervised_learning_models.types import ClassificationPredictionType
import graphene



class Query(graphene.ObjectType):
    """ Models query """
    supervised_learning_prediction = graphene.Field(ClassificationPredictionType, model=graphene.String())
    
    """ Models query resolver """
    def resolve_supervised_learning_prediction(self, info, model):
        
        # Nothing here yet.
        return ClassificationPredictionType(result = None)
   