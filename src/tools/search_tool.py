from langchain.tools import tool
from db.weaviate_client import vector_store

@tool
def search_user_guide(query: str) -> str:
    """Search user guide documents using semantic similarity."""
    docs = vector_store.similarity_search(query, k=3)
    return "\n\n".join([doc.page_content for doc in docs])