from youtube_transcript_api import YouTubeTranscriptApi
from functions import function
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document


def save_video_in_vectordb(video_id):

    print("Starting ingestion for Video ID :", video_id)

    GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
    index_name="yt-chat"

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GEMINI_API_KEY
    )

    yt_api = YouTubeTranscriptApi()

    print("Fetching Transcript...")
    transcript = yt_api.fetch(video_id, languages=['hi'])

    print("Combining Text...")
    full_text = " ".join(line.text for line in transcript if line.text)

    print("Translating Hindi to English...")
    eng_text = function.hindi_to_english(full_text)

    print("Chunking Text...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(eng_text)

    print("Creating Documents...")
    docs = [Document(page_content=chunk) for chunk in chunks]

    print("Storing Embeddings to Pinecone...")
    PineconeVectorStore.from_documents(
        docs,
        embeddings,
        index_name=index_name
    )

    print("Embedding Stored Successfully")
