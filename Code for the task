
# Importing neccessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from textblob import TextBlob

# Downloading stopwords and punkt
nltk.download('stopwords')
nltk.download('punkt')

# Loading the dataset
file_path = 'twitter_training.csv'
data = pd.read_csv(file_path)

# Renaming columns for better understanding
data.columns = ['ID', 'Topic', 'Sentiment', 'Tweet']

# Counting the number of instances for each sentiment
sentiment_counts = data['Sentiment'].value_counts()

# Plotting the sentiment distribution
plt.figure(figsize=(8, 5))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='viridis')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# Function to clean the tweet text
def preprocess_tweet(tweet):
    if not isinstance(tweet, str):
        return ""
    # Removing URLs, mentions, and hashtags
    tweet = re.sub(r'http\S+|www\S+|https\S+|@\S+|#\S+', '', tweet, flags=re.MULTILINE)
    # Removing special characters and numbers
    tweet = re.sub(r'\W', ' ', tweet)
    # Removing single characters
    tweet = re.sub(r'\s+[a-zA-Z]\s+', ' ', tweet)
    # Removing multiple spaces
    tweet = re.sub(r'\s+', ' ', tweet, flags=re.I)
    # Converting to lowercase
    tweet = tweet.lower()
    # Removing stopwords
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(tweet)
    filtered_tweet = ' '.join([word for word in word_tokens if word not in stop_words])
    return filtered_tweet

# Applying the preprocessing function to the tweet column
data['Cleaned_Tweet'] = data['Tweet'].apply(preprocess_tweet)

# Displayingthe first few rows of the cleaned dataset
print(data[['Tweet', 'Cleaned_Tweet']].head())

# Function to get the sentiment score
def get_sentiment(tweet):
    analysis = TextBlob(tweet)
    # Classifying sentiment
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'

# Applying the function to the cleaned tweet column
data['Predicted_Sentiment'] = data['Cleaned_Tweet'].apply(get_sentiment)

# Displaying the first few rows of the dataset with predicted sentiments
print(data[['Cleaned_Tweet', 'Predicted_Sentiment']].head())

# Counting the number of instances for each predicted sentiment
predicted_sentiment_counts = data['Predicted_Sentiment'].value_counts()

# Plotting the predicted sentiment distribution
plt.figure(figsize=(8, 5))
sns.barplot(x=predicted_sentiment_counts.index, y=predicted_sentiment_counts.values, palette='viridis')
plt.title('Predicted Sentiment Distribution')
plt.xlabel('Predicted Sentiment')
plt.ylabel('Count')
plt.show()
