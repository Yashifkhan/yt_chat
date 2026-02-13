from youtube_transcript_api import YouTubeTranscriptApi
from functions import function
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

GEMINI_API_KEY="AIzaSyAH0gobl4Bv_rQgCcfyyLrw-thTmNQt26o"
index_name="yt-chat"

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GEMINI_API_KEY
)

video_id="dfPwdJGRmdc"
yt_api=YouTubeTranscriptApi()

transcript = yt_api.fetch(video_id,languages=['hi'])

full_text = " ".join(line.text for line in transcript if line.text)
eng_text=function.hindi_to_english(full_text)

splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks=splitter.split_text(eng_text)

docs=[Document(page_content=chunk) for chunk in chunks]

PineconeVectorStore.from_documents(
        docs,
        embeddings,
        index_name=index_name
)

print("Embedding Stored Successfully")
