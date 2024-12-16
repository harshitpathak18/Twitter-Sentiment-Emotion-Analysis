import re
import nltk
import string
import joblib
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# list of stopwords
stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
             'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
             'does', 'doing', 'down', 'during', 'each','few', 'for', 'from',
             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma', 'id'
             'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're','s', 'same', 
             'she', "shes", 'should', "shouldve",'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
             'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
             'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
             'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
             "youve", 'your', 'yours', 'yourself', 'yourselves', "ive", 'couldnt']



# Load the vectorizer using joblib
vectorizer = joblib.load('./models/twitter_sentiment_tfidf_vectorizer.pkl')
model = joblib.load("./models/twitter_sentiment_model.pkl")

def predict_sentiment(tweet):
    # lowering the tweet
    tweet = str.lower(tweet)

    # removing integers
    tweet = re.sub("[0-9]+", '', tweet)

    # removing punctuations
    punctuations_list = string.punctuation
    translator = str.maketrans('', '', punctuations_list)
    tweet = tweet.translate(translator)
    
    # tokenizing the words
    tweet = word_tokenize(tweet)

    # removing stopwords
    tweet = [word for word in tweet if word not in stopwordlist]

    # Apply Lemmatization
    lemmatizer = WordNetLemmatizer()
    tweet = " ".join([lemmatizer.lemmatize(word, pos='v') for word in tweet])

    # Pass the string inside a list
    transformed_text = vectorizer.transform([tweet])
    
    # If you want to see the transformed result, you can convert it to an array
    transformed_text_array = transformed_text.toarray()
    
    # Making Predictions
    predictions = model.predict(transformed_text_array)
    
    # Return the predictions
    return predictions
   



if __name__ == "__main__":
    tweet1 = "Hello there, My name is Harry and I love what you did with the sentiment analysis system"
    tweet2 = "Hello there, i do not like you"
    tweet3 = "You are the best person i have encountered in my life"
    tweet4 = "I just went to my job, nothing new is happening now"
    tweet5 = "Hey there how you doin man, you are crazy person. you are a failed person"


    for tweet in [tweet1, tweet2, tweet3, tweet4, tweet5]:
        print(f"Tweet- {tweet}")
        prediction = predict_sentiment(tweet)
        mp = {-1:"Negative", 1:"Positive", 0:"Neutral"}
        print("Prediction:", mp[prediction[0]])
        print("")