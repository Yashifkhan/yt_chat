from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

PINECONE_API_KEY="pcsk_2bbBTM_9116sXkdN3THjT46Ng41gKDwEUEJBtGujqpeWGwWDV53THbiyhVpnVTWJzWDGKk"
index_name="yt-chat"

pc=Pinecone(api_key=PINECONE_API_KEY)
index=pc.Index(index_name)

def get_retriever(embeddings):

    vector_store=PineconeVectorStore(
            index=index,
            embedding=embeddings
    )

    return vector_store
