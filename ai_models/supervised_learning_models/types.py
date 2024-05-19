import graphene

class SupervisedLearningModelsType(graphene.ObjectType):
    knn = graphene.String()
    naive_bayes = graphene.String()
    logistic_regression = graphene.String()
    svm = graphene.String()
    random_forest = graphene.String()

class SupervisedLearningPredictionType(graphene.ObjectType):
    result = graphene.Field(SupervisedLearningModelsType)