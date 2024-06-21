import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import inflect

def conditional_lowercase(word):
    # Lowercase words that start with an uppercase letter but are not completely uppercase
    if word[0].isupper() and not word.isupper():
        return word.lower()
    return word

# Define pre-processing function
def preprocessed_with_stopwords_with_lemming(text):
    # Replace "5G" or "5 G" with "5 G"
    text = re.sub(r'\b(5\s?)G\b', '5 G', text)
    # Initialize inflect engine
    p = inflect.engine()
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Replace numbers with words
    text = re.sub(r'\d+', lambda x: p.number_to_words(x.group()), text)
    # Remove punctuation except exclamation mark
    text = re.sub(r'[^\w\s!]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Lowercase words conditionally
    tokens = [conditional_lowercase(word) for word in tokens]
    # Join tokens back to string
    text = ' '.join(tokens)
    # Strip white space
    text = text.strip()
    return text


# Define pre-processing function
def preprocessed_without_stopwords_with_lemming(text):
    # Replace "5G" or "5 G" with "5 G"
    text = re.sub(r'\b(5\s?)G\b', '5 G', text)
    # Initialize inflect engine
    p = inflect.engine()
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Replace numbers with words
    text = re.sub(r'\d+', lambda x: p.number_to_words(x.group()), text)
    # Remove punctuation except exclamation mark
    text = re.sub(r'[^\w\s!]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stop words
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Lowercase words conditionally
    tokens = [conditional_lowercase(word) for word in tokens]    
    # Join tokens back to string
    text = ' '.join(tokens)
    # Strip white space
    text = text.strip()
    return text

# Define pre-processing function
def preprocessed_with_stopwords_without_lemming(text):
    # Replace "5G" or "5 G" with "5 G"
    text = re.sub(r'\b(5\s?)G\b', '5 G', text)    
    # Initialize inflect engine
    p = inflect.engine()
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Replace numbers with words
    text = re.sub(r'\d+', lambda x: p.number_to_words(x.group()), text)
    # Remove punctuation except exclamation mark
    text = re.sub(r'[^\w\s!]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Lowercase words conditionally
    tokens = [conditional_lowercase(word) for word in tokens]    
    # Join tokens back to string
    text = ' '.join(tokens)
    # Strip white space
    text = text.strip()
    return text

# Define pre-processing function
def preprocessed_without_stopwords_without_lemming(text):
    # Replace "5G" or "5 G" with "5 G"
    text = re.sub(r'\b(5\s?)G\b', '5 G', text)
    # Initialize inflect engine
    p = inflect.engine()
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Replace numbers with words
    text = re.sub(r'\d+', lambda x: p.number_to_words(x.group()), text)
    # Remove punctuation except exclamation mark
    text = re.sub(r'[^\w\s!]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Lowercase words conditionally
    tokens = [conditional_lowercase(word) for word in tokens]    
    # Remove stop words
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    # Join tokens back to string
    text = ' '.join(tokens)
    # Strip white space
    text = text.strip()
    return text