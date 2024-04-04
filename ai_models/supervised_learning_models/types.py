import graphene

class SupervisedLearningModelsType(graphene.ObjectType):
    knn = graphene.String()

class SupervisedLearningPredictionType(graphene.ObjectType):
    result = graphene.Field(SupervisedLearningModelsType)