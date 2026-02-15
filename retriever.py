from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
import os
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
index_name=os.getenv("PINECONE_INDEX")


pc=Pinecone(api_key=PINECONE_API_KEY)
index=pc.Index(index_name)

def get_retriever(embeddings):
    vector_store=PineconeVectorStore(
            index=index,
            embedding=embeddings
    )

    return vector_store
