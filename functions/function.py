from langchain_groq import ChatGroq
import re
from dotenv import load_dotenv
load_dotenv()

model =ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")

def hindi_to_english(text):
    model_response=model.invoke([
        {"role":"system","content":"you are a helpful assistant that can convert hindi to english"},
        {"role":"user","content":f"convert this hindi text to clear english : {text}"}])
    return model_response.content

def contains_youtube_url(text):
    youtube_regex = r"(https?://)?(www\.)?(youtube\.com|youtu\.be)/\S+"
    match = re.search(youtube_regex, text)
    return match is not None

def classify_user_query(query):

    if contains_youtube_url(query):
        return "NEW_VIDEO"
    else:
        return "EXISTING_VIDEO"
    
def extract_video_id(text):
    pattern = r"(?:v=|youtu\.be/|embed/|shorts/)([a-zA-Z0-9_-]{11})"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        return None
