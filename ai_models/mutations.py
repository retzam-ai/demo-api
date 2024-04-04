import graphene
from ai_models.supervised_learning_models.mutations import SupervisedLearningPredictionMutation

class Mutation(graphene.ObjectType):
    supervised_learning_prediction = SupervisedLearningPredictionMutation.Field()
    