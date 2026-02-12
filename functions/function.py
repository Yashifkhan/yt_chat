from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

model =ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")

def hindi_to_english(text):
    model_response=model.invoke([
        {"role":"system","content":"you are a helpful assistant that can convert hindi to english"},
        {"role":"user","content":f"convert this hindi text to clear english : {text}"}])
    return model_response.content


