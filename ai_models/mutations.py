import graphene
from ai_models.supervised_learning_models.mutations import SupervisedLearningPredictionClassificationMutation

class Mutation(graphene.ObjectType):
    supervised_learning_classification_prediction = SupervisedLearningPredictionClassificationMutation.Field()
    