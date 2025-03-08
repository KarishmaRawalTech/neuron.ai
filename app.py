from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from sentence_transformers import SentenceTransformer

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize FastAPI app
app = FastAPI()

# Load the pre-trained SentenceTransformer model
with open("similarity_model.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess(text):
    """Preprocesses input text by lowercasing, removing special characters, tokenizing, removing stopwords, and lemmatizing."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Request body structure
class TextSimilarityRequest(BaseModel):
    text1: str
    text2: str

# API endpoint
@app.post("/predict/")
def get_similarity(data: TextSimilarityRequest):
    """Takes in two text inputs, preprocesses them, converts them to embeddings, and computes similarity."""
    # Preprocess input texts
    text1_clean = preprocess(data.text1)
    text2_clean = preprocess(data.text2)
    
    # Convert texts to embeddings using SentenceTransformer
    embeddings_text1 = model.encode([text1_clean], convert_to_tensor=True)
    embeddings_text2 = model.encode([text2_clean], convert_to_tensor=True)
    
    # Compute similarity score
    similarity_score = cosine_similarity(
        [embeddings_text1.cpu().detach().numpy()[0]],
        [embeddings_text2.cpu().detach().numpy()[0]]
    )[0][0]
    
    return {"similarity score": round(float(similarity_score), 4)}
