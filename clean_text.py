import re, nltk, pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

phishing_lexicon = [
    'verify', 'account', 'password', 'suspended', 'click', 'security',
    'login', 'confirm', 'urgent', 'bank', 'limited', 'access', 'update', 'failure'
]

def clean_text(text):
    if pd.isna(text):
        return ''

    text = str(text).lower()
    text = re.sub(r'(?!(http[s]?://|www\.|@|£|\$|€))[^a-zA-Z\s]', '', text)
    text = re.sub(r'(\.{2,})', ' DOTSEQ ', text)

    tokens = []
    for word in text.split():
        if word not in stop_words:
            if 'http' in word or 'www' in word:
                tokens.append('URL_TOKEN')
            elif '@' in word:
                tokens.append('EMAIL_TOKEN')
            else:
                tokens.append(stemmer.stem(word))

    return ' '.join(tokens)

def extract_features(text_list, tfidf=None, fit=False):
    df = pd.DataFrame({'text_combined': text_list})

    df['num_links'] = df['text_combined'].str.count(r'http[s]?://')
    df['has_attachment'] = df['text_combined'].str.contains(r'attach|enclosure', case=False).astype(int)
    df['urgency_score'] = df['text_combined'].str.count(r'urgent|immediate|action required', flags=re.IGNORECASE)

    for word in phishing_lexicon:
        df[f'kw_{word}'] = df['text_combined'].str.count(word, flags=re.IGNORECASE)

    df['cleaned_text'] = df['text_combined'].apply(clean_text)

    if tfidf is None:
        tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 3), stop_words='english')

    if fit:
        X_text = tfidf.fit_transform(df['cleaned_text'])
    else:
        X_text = tfidf.transform(df['cleaned_text'])
    
    extra_feats = df[['num_links', 'has_attachment', 'urgency_score'] + [f'kw_{w}' for w in phishing_lexicon]].values
    X_final = hstack([X_text, extra_feats])
    return X_final, tfidf