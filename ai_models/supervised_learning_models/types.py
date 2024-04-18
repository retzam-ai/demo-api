import graphene

class SupervisedLearningModelsType(graphene.ObjectType):
    knn = graphene.String()
    naive_bayes = graphene.String()

class SupervisedLearningPredictionType(graphene.ObjectType):
    result = graphene.Field(SupervisedLearningModelsType)