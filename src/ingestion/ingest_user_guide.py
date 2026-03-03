import click
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from db.weaviate_client import vector_store

@click.command()
@click.argument("file_path")
def ingest(file_path):

    loader = TextLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)

    vector_store.add_documents(chunks)

if __name__ == "__main__":
    ingest()