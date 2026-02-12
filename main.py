from youtube_transcript_api import YouTubeTranscriptApi
from functions import function
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.generativeai as genai
import os

from dotenv import load_dotenv
load_dotenv()

model =ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")
genai.configure(api_key=os.getenv("GEMINI_KEY"))

def embed_text(text):
    result = genai.embed_content(
        { model: "text-embedding-004" },
        content=text
    )
    return result["embedding"]

video_id = "dfPwdJGRmdc"  
yt_api= YouTubeTranscriptApi()
transcript =yt_api.fetch(video_id, languages=['hi'])
full_text = " ".join(line.text for line in transcript if line.text)
eng_text=function.hindi_to_english(full_text)

splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chuncks=splitter.split_text(eng_text)

print("Total chuncks : ",len(chuncks))
embedding = embed_text(chuncks[0])

print("Embedding length:", len(embedding))

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the answer is not in the context, say you don't know.

Transcript Context:
{context}
"""),
    ("human", "Question: {question}")
])


parser=StrOutputParser()
chain = prompt | model | parser

# response = chain.invoke({
#     "context": eng_text,
#     "question": "What is the video about?"
# })
# print("LLm response : ",response)

