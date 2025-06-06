import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Optional: Only needed once if used interactively
nltk.download('stopwords')

# Initialize stemmer and stopwords (move these outside the function for efficiency)
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove non-letter characters (keeps spaces)
    text = re.sub(r'[^a-z\s]', '', text)

    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize and remove stopwords
    tokens = [word for word in text.split() if word not in stop_words]

    # Stemming
    stemmed = [stemmer.stem(word) for word in tokens]

    return ' '.join(stemmed)
