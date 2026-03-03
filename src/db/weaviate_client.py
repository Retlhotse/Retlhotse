import weaviate
from langchain_weaviate import WeaviateVectorStore
from langchain_openai import OpenAIEmbeddings
from config import settings

client = weaviate.connect_to_local(
    host=settings.WEAVIATE_URL
)

embeddings = OpenAIEmbeddings()

vector_store = WeaviateVectorStore(
    client=client,
    index_name="UserGuide",
    embedding=embeddings
)