import graphene

class ClassificationModelsType(graphene.ObjectType):
    knn = graphene.String()
    naive_bayes = graphene.String()
    logistic_regression = graphene.String()
    svm = graphene.String()
    random_forest = graphene.String()

class ClassificationPredictionType(graphene.ObjectType):
    result = graphene.Field(ClassificationModelsType)
    
class RegressionModelsType(graphene.ObjectType):
    simple_linear_regression = graphene.String()
    multiple_linear_regression = graphene.String()
    
class RegressionPredictionType(graphene.ObjectType):
    result = graphene.Field(RegressionModelsType)
   