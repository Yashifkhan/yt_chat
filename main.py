# from youtube_transcript_api import YouTubeTranscriptApi
# from functions import function
# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from pinecone import Pinecone
# from langchain_pinecone import PineconeVectorStore
# from langchain_core.documents import Document
# import os

# from dotenv import load_dotenv
# load_dotenv()

# os.environ["GOOGLE_API_VERSION"] = "v1"

# model =ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")

# GEMINI_API_KEY="AIzaSyAH0gobl4Bv_rQgCcfyyLrw-thTmNQt26o"
# PINECONE_API_KEY="pcsk_2bbBTM_9116sXkdN3THjT46Ng41gKDwEUEJBtGujqpeWGwWDV53THbiyhVpnVTWJzWDGKk"

# GEMINI_API_KEY="GEMINI_API_KEY"
# PINECONE_API_KEY="PINECONE_API_KEY"
# index_name = "yt-chat"

# embeddings = GoogleGenerativeAIEmbeddings(
#     model="models/gemini-embedding-001",
#     google_api_key=GEMINI_API_KEY
# )
# pc = Pinecone(
#     api_key=PINECONE_API_KEY
# )

# video_id = "dfPwdJGRmdc"  
# yt_api= YouTubeTranscriptApi()
# transcript =yt_api.fetch(video_id, languages=['hi'])
# full_text = " ".join(line.text for line in transcript if line.text)
# eng_text=function.hindi_to_english(full_text)

# splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
# chuncks=splitter.split_text(eng_text)

# docs = [Document(page_content=chunk) for chunk in chuncks]


# vector_store = PineconeVectorStore.from_documents(
#     docs,
#     embeddings,
#     index_name=index_name
# )

# user_query = "What is the video about"
# docs = vector_store.similarity_search(
#          user_query,
#          k=3
# )

# prompt = ChatPromptTemplate.from_messages([
#     ("system", """You are a helpful assistant.
# Answer ONLY from the provided transcript context.
# If the answer is not in the context, say you don't know.

# Transcript Context:
# {context}
# """),
#     ("human", "Question: {question}")
# ])

# parser=StrOutputParser()
# chain = prompt | model | parser

# response = chain.invoke({
#     "context" : "\n".join([doc.page_content for doc in docs]),
#     "question": user_query
# })
# print("LLm response with vc db : ",response)




from retriever import get_retriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from functions import function
from ingest import save_video_in_vectordb
import os

load_dotenv()

GEMINI_API_KEY="AIzaSyAH0gobl4Bv_rQgCcfyyLrw-thTmNQt26o"
PINECONE_API_KEY="pcsk_2bbBTM_9116sXkdN3THjT46Ng41gKDwEUEJBtGujqpeWGwWDV53THbiyhVpnVTWJzWDGKk"

model = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct"
)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GEMINI_API_KEY
)
vector_store = get_retriever(embeddings)

# -------------------- User Query --------------------
# https://www.youtube.com/watch?v=6QkH6QDKZ3g&list=RD6QkH6QDKZ3g&start_radio=1
user_query = """ What is the video about? 
"""
classifyedQuery=function.classify_user_query(user_query)
if classifyedQuery=="NEW_VIDEO":
    vedio_id=function.extract_video_id(user_query)
    print("This is a new video, need to ingest and create vector db",vedio_id)
    save_video_in_vectordb(vedio_id)
    
    docs = vector_store.similarity_search(
        user_query,
        k=3
    )
    for i, doc in enumerate(docs):
        prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the answer is not in the context, say you don't know.
        
        Transcript Context:
        {context}
        """),
            ("human", "Question: {question}")
        ])
         
        parser = StrOutputParser()
        chain = prompt | model | parser
        
        # -------------------- Final RAG Response --------------------
        response = chain.invoke({
            "context": "\n".join([doc.page_content for doc in docs]),
            "question": user_query
        })
        print("llm response",response)
        break


elif classifyedQuery=="EXISTING_VIDEO":
    print("Existing video, fetching from vector db")
    docs = vector_store.similarity_search(
        user_query,
        k=3
    )
    for i, doc in enumerate(docs):
        prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant.
    Answer ONLY from the provided transcript context.
    If the answer is not in the context, say you don't know.
    
    Transcript Context:
    {context}
    """),
        ("human", "Question: {question}")
    ])
         
        parser = StrOutputParser()
        chain = prompt | model | parser
        
        # -------------------- Final RAG Response --------------------
        response = chain.invoke({
            "context": "\n".join([doc.page_content for doc in docs]),
            "question": user_query
        })
        print("llm response",response)
        break
