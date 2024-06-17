import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import inflect

# Define pre-processing function
def preprocessed_with_stopwords_with_lemming(text):
    # Initialize inflect engine
    p = inflect.engine()
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Replace numbers with words
    text = re.sub(r'\d+', lambda x: p.number_to_words(x.group()), text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join tokens back to string
    text = ' '.join(tokens)
    # Strip white space
    text = text.strip()
    return text


# Define pre-processing function
def preprocessed_without_stopwords_with_lemming(text):
    # Initialize inflect engine
    p = inflect.engine()
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Replace numbers with words
    text = re.sub(r'\d+', lambda x: p.number_to_words(x.group()), text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stop words
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join tokens back to string
    text = ' '.join(tokens)
    # Strip white space
    text = text.strip()
    return text

# Define pre-processing function
def preprocessed_with_stopwords_without_lemming(text):
    # Initialize inflect engine
    p = inflect.engine()
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Replace numbers with words
    text = re.sub(r'\d+', lambda x: p.number_to_words(x.group()), text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Join tokens back to string
    text = ' '.join(tokens)
    # Strip white space
    text = text.strip()
    return text

# Define pre-processing function
def preprocessed_without_stopwords_without_lemming(text):
    # Initialize inflect engine
    p = inflect.engine()
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Replace numbers with words
    text = re.sub(r'\d+', lambda x: p.number_to_words(x.group()), text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stop words
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    # Join tokens back to string
    text = ' '.join(tokens)
    # Strip white space
    text = text.strip()
    return text