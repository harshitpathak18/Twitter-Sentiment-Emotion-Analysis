import re
import nltk
import string
import joblib
from pydantic import BaseModel
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# Initialize FastAPI app
app = FastAPI()



# Allow CORS for frontend (you can specify allowed origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # "*" allows all origins, for production specify only your frontend URL
    allow_credentials=True,
    allow_methods=["POST"],  # Allow all methods (POST, GET, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
)




# Load the vectorizer and model
try:
    tfidf_vectorizer = joblib.load('./models/twitter_sentiment_tfidf_vectorizer.pkl')
    model = joblib.load("./models/twitter_sentiment_model.pkl")
except Exception as e:
    raise RuntimeError(f"Error loading models: {e}")

# List of stopwords
stopwordlist = [
    'a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
    'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
    'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
    'does', 'doing', 'down', 'during', 'each','few', 'for', 'from',
    'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
    'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
    'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma', 'id',
    'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
    'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're','s', 'same', 
    'she', "shes", 'should', "shouldve",'so', 'some', 'such',
    't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
    'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
    'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
    'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
    'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
    "youve", 'your', 'yours', 'yourself', 'yourselves', "ive", 'couldnt'
]

# Define input data model
class Tweet(BaseModel):
    text: str

# Sentiment prediction function
def predict_sentiment(tweet):
    try:
        # Lowercasing
        tweet = str.lower(tweet)

        # Removing integers
        tweet = re.sub("[0-9]+", '', tweet)

        # Removing punctuations
        punctuations_list = string.punctuation
        translator = str.maketrans('', '', punctuations_list)
        tweet = tweet.translate(translator)
        
        # Tokenizing the words
        tweet = word_tokenize(tweet)

        # Removing stopwords
        tweet = [word for word in tweet if word not in stopwordlist]

        # Apply Lemmatization
        lemmatizer = WordNetLemmatizer()
        tweet = " ".join([lemmatizer.lemmatize(word, pos='v') for word in tweet])

        # Transform using TF-IDF vectorizer
        transformed_text = tfidf_vectorizer.transform([tweet])

        # Predict sentiment
        prediction = model.predict(transformed_text.toarray())

        # Ensure the result is a Python int
        return int(prediction[0])

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing tweet: {e}")

# Define the API endpoint
@app.post("/tweet-sentiment")
async def get_sentiment(tweet: Tweet):
    print(tweet)
    sentiment = predict_sentiment(tweet.text)
    return {"sentiment": sentiment}

