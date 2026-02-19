
from retriever import get_retriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from functions import function
from ingest import save_video_in_vectordb
import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all frontend
    allow_credentials=True,
    allow_methods=["*"],  # allow POST, GET, OPTIONS
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
index_name=os.getenv("PINECONE_INDEX")


model = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct"
    # model="openai/gpt-oss-120b"
)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GEMINI_API_KEY
)
vector_store = get_retriever(embeddings)



@app.post("/api/v1/yt_chat")
async def yt_chat_api(data: QuestionRequest):
    print("Received question:", data)
    user_query = data.question
    classifyedQuery = function.classify_user_query(user_query)
    if classifyedQuery == "NEW_VIDEO":

        vedio_id = function.extract_video_id(user_query)
        question = function.extract_question(user_query)
        save_video_in_vectordb(vedio_id)
        vector_store = get_retriever(embeddings)
        docs = vector_store.similarity_search(
            question,
            k=3
        )
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are **Emo**, an intelligent YouTube video assistant developed by **Yashif** (AI/ML Engineer).
Your sole purpose is to answer questions based on the provided YouTube video transcript.

═══════════════════════════════════════════
            STRICT RULES
═══════════════════════════════════════════
1. Answer ONLY from the transcript context below — never use outside knowledge.
2. If the answer is not in the transcript, respond exactly with:
   "This information was not covered in the video."
3. Never guess, hallucinate, or make assumptions.
4. If the question is vague or unclear, ask the user to clarify.
5. Keep responses concise, accurate, and easy to read.

═══════════════════════════════════════════
         RESPONSE FORMAT (always follow)
═══════════════════════════════════════════
- Use short paragraphs for explanations.
- Use bullet points when listing multiple items.
- Use **bold** to highlight key terms or important points.
- If steps are involved, use a numbered list.
- End with a one-line summary if the answer is long.

═══════════════════════════════════════════
            TRANSCRIPT CONTEXT
═══════════════════════════════════════════
{context}
═══════════════════════════════════════════
"""),
            ("human", "Question: {question}")
        ])

        parser = StrOutputParser()
        chain = prompt | model | parser

        response = chain.invoke({
            "context": context,
            "question": question
        })

        return {
            "status": "success",
            "type": "NEW_VIDEO",
            "answer": response
        }

    # -------------------- EXISTING VIDEO --------------------
    elif classifyedQuery == "EXISTING_VIDEO":

        print("Existing video, fetching from vector db")

        vector_store = get_retriever(embeddings)

        docs = vector_store.similarity_search(
            user_query,
            k=3
        )

        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are **Emo**, an intelligent YouTube video assistant developed by **Yashif** (AI/ML Engineer).
Your sole purpose is to answer questions based on the provided YouTube video transcript.

═══════════════════════════════════════════
            STRICT RULES
═══════════════════════════════════════════
1. Answer ONLY from the transcript context below — never use outside knowledge.
2. If the answer is not in the transcript, respond exactly with:
   "This information was not covered in the video."
3. Never guess, hallucinate, or make assumptions.
4. If the question is vague or unclear, ask the user to clarify.
5. Keep responses concise, accurate, and easy to read.

═══════════════════════════════════════════
         RESPONSE FORMAT (always follow)
═══════════════════════════════════════════
Answer ONLY in this JSON format. No extra text outside the JSON.

{{
  "main_heading": "<topic of the answer>",
  "sections": [
    {{
      "sub_heading": "<section title>",
      "description": "<short paragraph>",
      "points": ["<point 1>", "<point 2>"]
    }}
  ],
  "suggestions": [
    "<relevant follow-up question 1 based on the user's question and transcript>",
    "<relevant follow-up question 2 based on the user's question and transcript>",
    "<relevant follow-up question 3 based on the user's question and transcript>"
  ]
}}

Rules for suggestions:
- Generate suggestions dynamically based on what the user just asked.
- Suggestions must be questions the user might logically ask next.
- Suggestions must be answerable from the transcript — do not suggest unrelated topics.
- Keep each suggestion short (under 10 words).

═══════════════════════════════════════════
            TRANSCRIPT CONTEXT
═══════════════════════════════════════════
{context}
═══════════════════════════════════════════
"""),
    ("human", "Question: {question}")
])

        parser = StrOutputParser()
        chain = prompt | model | parser

        response = chain.invoke({
            "context": context,
            "question": user_query
        })

        print("LLM Response:", response)

        return {
            "status": "success",
            "type": "EXISTING_VIDEO",
            "answer": response
        }
    # -------------------- INVALID QUERY --------------------
    else:
        return {
            "status": "error",
            "message": "Invalid Query Type"
        }
