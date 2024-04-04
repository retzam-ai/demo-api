import graphene
from ai_models.queries import Query
from ai_models.mutations import Mutation

schema = graphene.Schema(query=Query, mutation=Mutation)